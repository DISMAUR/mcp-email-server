"""Bounded protocol adapters over aioimaplib and aiosmtplib.

The small IMAP subclass owns receive limits, safe logging and APPEND continuation
admission. Parsing/framing and connection setup stay with the existing clients.
No dependency-global command table or logger settings are modified.
"""

from __future__ import annotations

import asyncio
import ssl
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from typing import cast

import aioimaplib
import aiosmtplib
from aioimaplib import aioimaplib as imaplib
from aiosmtplib.errors import SMTPResponseException

from mcp_email_server.adapters.authority import LocalAccountResolution, resolve_local_account
from mcp_email_server.application.protocol import (
    IMAP_COMMANDS,
    MAX_RECEIVED_BYTES,
    MAX_RESPONSE_BLOCKS,
    ParsedCommand,
    ProtocolFailure,
    ProtocolName,
    ProtocolSession,
    ProtocolStep,
    ResponseBlock,
    mailbox_text,
)
from mcp_email_server.config import EmailSettings
from mcp_email_server.emails.classic import EmailClient, _create_ssl_context, _imap_login, _quote_mailbox
from mcp_email_server.metadata_index import MetadataIndex


def require_protocol_access(resolved: LocalAccountResolution, protocol: ProtocolName) -> None:
    account, settings = resolved.account, resolved.settings
    if not account.allow_protocol_commands:
        raise ProtocolFailure(
            "protocol_disabled: a user must enable allow_protocol_commands in configuration, CLI or UI; prefer focused tools"
        )
    grants = account.allowed_mutations if account.allowed_mutations is not None else settings.allowed_mutations
    if protocol == "imap":
        if (
            not {"organize", "delete", "append"}.issubset(grants)
            or settings.allowed_senders
            or not settings.enable_attachment_content
        ):
            raise ProtocolFailure(
                "policy_conflict: IMAP requires organize/delete/append, no sender filter and attachment content enabled; use focused tools or ask the user to configure access"
            )
    elif account.outgoing is None or "send" not in grants or "*" not in settings.allowed_recipients:
        raise ProtocolFailure(
            "policy_conflict: SMTP requires outgoing configuration, send permission and explicit * allowed_recipients; an empty list denies access"
        )


def _same_authority(left: LocalAccountResolution, right: LocalAccountResolution) -> bool:
    # Managed credentials remain unread during guards; revisions cover rotation.
    # Legacy settings carry SecretStr values, compared in memory, never serialized.
    return (
        left.mode == right.mode
        and left.authority_revision == right.authority_revision
        and left.settings.db_location == right.settings.db_location
        and left.account == right.account
        and left.settings.allowed_mutations == right.settings.allowed_mutations
        and left.settings.allowed_senders == right.settings.allowed_senders
        and left.settings.allowed_recipients == right.settings.allowed_recipients
        and left.settings.enable_attachment_content == right.settings.enable_attachment_content
    )


class BoundedImapProtocol(imaplib.IMAP4ClientProtocol):
    """One command at a time, retaining all replies without wire-content logs."""

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        super().__init__(loop)
        self.received = 0
        self.blocks = 0
        self.before_write: Callable[[], None] | None = None
        self.failure: ProtocolFailure | None = None

    def _pending(self) -> list[imaplib.Command]:
        pending = cast(list[imaplib.Command], list(self.pending_async_commands.values()))
        if self.pending_sync_command is not None:
            pending.append(self.pending_sync_command)
        return pending

    def abort(self, message: str) -> None:
        self.failure = ProtocolFailure(message)
        for command in self._pending():
            command._exception = self.failure  # pyright: ignore[reportAttributeAccessIssue] - dependency narrows Exception incorrectly
            command.close(b"", "BAD")
        if self.transport is not None:
            self.transport.close()

    def data_received(self, d: bytes) -> None:
        data = d
        if self.failure is not None:
            return
        self.received += len(data)
        if self.received > MAX_RECEIVED_BYTES:
            self.abort("receive_limit: server response exceeded 512 KiB; command completion may be unknown")
            return
        try:
            self._handle_responses(
                self.incomplete_line + data, self._handle_line, cast(imaplib.Command, self.current_command)
            )
            self.incomplete_line = b""
            self.current_command = None
        except imaplib.IncompleteRead as incomplete:
            self.current_command = incomplete.cmd
            self.incomplete_line = incomplete.data
        except Exception:
            self.abort("protocol_interrupted: invalid response or authority changed before payload")

    def _handle_line(self, line: bytes, current_cmd: imaplib.Command | None) -> imaplib.Command | None:
        self.blocks += 1
        if self.blocks > MAX_RESPONSE_BLOCKS:
            raise ProtocolFailure("receive_limit: too many response blocks")
        if self.state == imaplib.CONNECTED:
            return super()._handle_line(line, cast(imaplib.Command, current_cmd))
        if imaplib.tagged_status_response_re.match(line):
            self._response_done(line)
        elif current_cmd is not None:
            current_cmd.append_to_resp(line)
            return current_cmd
        elif line.startswith(b"* "):
            if line.upper().startswith(b"* BYE"):
                raise ProtocolFailure("connection_closed: server ended the session")
            pending = self._pending()
            if pending:
                pending[0].append_to_resp(line[2:])
                return pending[0]
        elif line.startswith(b"+"):
            self._continuation(line)
        elif line:
            raise ProtocolFailure("protocol_interrupted: unexpected response")
        return None

    def _response_done(self, line: bytes) -> None:
        tag, _, response = line.partition(b" ")
        matching = [command for command in self._pending() if command.tag.encode() == tag]
        if len(matching) != 1:
            raise ProtocolFailure("protocol_interrupted: unexpected completion tag")
        command = matching[0]
        if command is self.pending_sync_command:
            self.pending_sync_command = None
        else:
            self.pending_async_commands.pop(command.untagged_resp_name)
        code, _, text = response.partition(b" ")
        command.close(text, code.decode("ascii"))

    def send(self, line: str, scrub: str | None = None) -> None:
        if self.before_write is not None:
            self.before_write()
        if self.transport is None or self.transport.is_closing():
            raise ProtocolFailure("connection_closed: no command sent")
        cast(asyncio.WriteTransport, self.transport).write((line + "\r\n").encode("utf-8"))

    def _continuation(self, line: bytes) -> None:
        if self.pending_sync_command is None or self.pending_sync_command.name != "APPEND" or self.literal_data is None:
            raise ProtocolFailure("unsupported_continuation: only ordinary APPEND literals are supported")
        if self.before_write is not None:
            self.before_write()
        cast(asyncio.WriteTransport, self.transport).write(self.literal_data + b"\r\n")
        self.literal_data = None

    def connection_lost(self, exc: Exception | None) -> None:
        self.abort("connection_closed: command completion may be unknown")


class BoundedImap(aioimaplib.IMAP4):
    protocol: BoundedImapProtocol

    def create_client(
        self,
        host: str,
        port: int,
        loop: asyncio.AbstractEventLoop | None,
        conn_lost_cb=None,
        ssl_context: ssl.SSLContext | None = None,
    ) -> None:
        local_loop = loop or asyncio.get_running_loop()
        self.protocol = BoundedImapProtocol(local_loop)
        self._client_task = local_loop.create_task(
            local_loop.create_connection(lambda: self.protocol, host, port, ssl=ssl_context)
        )


class LocalProtocolSession:
    def __init__(self, initial: LocalAccountResolution, protocol: ProtocolName) -> None:
        self.initial = initial
        self.protocol_name: ProtocolName = protocol
        self.imap: BoundedImap | None = None
        self.smtp: aiosmtplib.SMTP | None = None
        self.sent = False
        self.received = 0
        self.index = MetadataIndex(Path(initial.settings.db_location), initial.mode)
        self.operational_account_id: str | None = None

    def guard(self) -> None:
        try:
            current = resolve_local_account(
                self.initial.account.account_name, expected_mode=self.initial.mode, fresh_legacy=True
            )
            require_protocol_access(current, self.protocol_name)
        except ProtocolFailure:
            raise
        except Exception:
            raise ProtocolFailure("authority_unavailable: account or policy cannot be revalidated") from None
        if not _same_authority(self.initial, current):
            raise ProtocolFailure("authority_changed: account or policy changed; connection discarded")

    def before_write(self) -> None:
        self.guard()
        self.sent = True

    def reset_step(self) -> None:
        self.sent = False

    async def invalidate(self) -> None:
        if self.operational_account_id is None:
            self.operational_account_id = await asyncio.to_thread(
                self.index.resolve_operational_account, self.initial.account
            )
        await asyncio.to_thread(self.index.invalidate_account, self.operational_account_id)

    async def open(self) -> None:  # noqa: C901 - selected-role setup for two transports
        self.guard()
        role = "incoming" if self.protocol_name == "imap" else "outgoing"
        resolved = resolve_local_account(
            self.initial.account.account_name, roles=(role,), expected_mode=self.initial.mode, fresh_legacy=True
        )
        # Compare the same credential-free managed projection on either side of
        # secret resolution; the revision returned with credentials binds both.
        if resolved.mode == "managed" and resolved.authority_revision != self.initial.authority_revision:
            raise ProtocolFailure("authority_changed: account changed during credential resolution")
        if resolved.mode != "managed":
            snapshot_account = resolved.settings.get_account(self.initial.account.account_name)
            if not isinstance(snapshot_account, EmailSettings) or not _same_authority(
                self.initial, replace(resolved, account=snapshot_account)
            ):
                raise ProtocolFailure("authority_changed: legacy account changed during credential resolution")
        self.guard()
        server = resolved.account.incoming if role == "incoming" else resolved.account.outgoing
        if server is None:
            raise ProtocolFailure("policy_conflict: outgoing configuration is required")
        if self.protocol_name == "imap":
            context = (
                (_create_ssl_context(server.verify_ssl) or ssl.create_default_context()) if server.use_ssl else None
            )
            self.imap = BoundedImap(server.host, server.port, ssl_context=cast(ssl.SSLContext, context))
            await EmailClient._prepare_imap_connection(self.imap, server)
            await _imap_login(self.imap, server.user_name, server.password.get_secret_value())
            if "UTF8=ONLY" in self.imap.protocol.capabilities:
                raise ProtocolFailure(
                    "unsupported_server: UTF8=ONLY sessions are not supported by this ordinary-literal executor"
                )
            self.imap.protocol.before_write = self.before_write
        else:
            self.smtp = aiosmtplib.SMTP(
                hostname=server.host,
                port=server.port,
                start_tls=server.start_ssl,
                use_tls=server.use_ssl,
                tls_context=_create_ssl_context(server.verify_ssl),
            )
            await self.smtp.connect()
            await self.smtp.login(server.user_name, server.password.get_secret_value())
            protocol = self.smtp.protocol
            if protocol is None:
                raise ProtocolFailure("connection_closed: SMTP setup did not complete")
            write, receive = protocol.write, protocol.data_received

            def guarded_write(data: bytes) -> None:
                self.before_write()  # Also runs between DATA's 354 and its body.
                write(data)

            def bounded_receive(data: bytes) -> None:
                self.received += len(data)
                if self.received > MAX_RECEIVED_BYTES:
                    self.close()
                    return
                receive(data)

            protocol.write = guarded_write
            protocol.data_received = bounded_receive

    def close(self) -> None:
        # Closing transports cannot implicitly expunge an IMAP mailbox.
        if self.imap is not None:
            for command in self.imap.protocol._pending():
                command._timer.cancel()
            if self.imap.protocol.transport is not None:
                self.imap.protocol.transport.close()
            if not self.imap._client_task.done():
                self.imap._client_task.cancel()
        if self.smtp is not None:
            protocol = self.smtp.protocol
            if protocol is not None and protocol._response_waiter is not None:
                # A guard can stop between DATA's 354 and its body, leaving a
                # reply waiter with no reader. Consume a later disconnect error.
                waiter = protocol._response_waiter
                waiter.add_done_callback(lambda future: None if future.cancelled() else future.exception())
            self.smtp.close()

    async def execute(self, command: ParsedCommand) -> ProtocolStep:
        self.guard()
        if self.imap is not None:
            return await self._imap_execute(command)
        if self.smtp is None:
            raise ProtocolFailure("connection_closed: no authenticated session")
        try:
            async with asyncio.timeout(self.smtp.timeout):
                if command.name == "DATA":
                    response = await self.smtp.data(command.data or b"")
                else:
                    response = await self.smtp.execute_command(
                        command.name.encode("ascii"), *(arg.encode("utf-8") for arg in command.args)
                    )
            code, message = response.code, response.message
        except SMTPResponseException as exc:
            if not 400 <= exc.code < 600:
                raise ProtocolFailure(
                    "invalid_response: SMTP completion could not be parsed; do not replay automatically"
                ) from None
            code, message = exc.code, exc.message
        return ProtocolStep(
            index=0,
            status="ok" if 200 <= code < 300 else "error",
            code=str(code),
            responses=[ResponseBlock.from_bytes(message.encode("utf-8", "surrogateescape"))],
        )

    async def _imap_execute(self, command: ParsedCommand) -> ProtocolStep:
        if self.imap is None:
            raise ProtocolFailure("connection_closed: IMAP is unavailable")
        protocol = self.imap.protocol
        syntax = IMAP_COMMANDS[command.name]
        if syntax.selected and protocol.state != imaplib.SELECTED:
            raise ProtocolFailure("invalid_state: SELECT or EXAMINE a mailbox earlier in this call")
        if command.name == "MOVE" and "MOVE" not in protocol.capabilities:
            raise ProtocolFailure("unsupported_capability: server does not advertise MOVE; no fallback attempted")
        if command.uid and command.name == "EXPUNGE" and "UIDPLUS" not in protocol.capabilities:
            raise ProtocolFailure("unsupported_capability: UID EXPUNGE requires UIDPLUS")
        args = list(command.args)
        for index in syntax.mailboxes:
            args[index] = _quote_mailbox(mailbox_text(args[index]))
        async with asyncio.timeout(self.imap.timeout):
            if command.name == "APPEND":
                options = args[1:]
                flags = options.pop(0) if options and options[0].startswith("(") else None
                response = await self.imap.append(
                    command.data or b"", mailbox=args[0], flags=cast(str, flags), date=options[0] if options else None
                )
            else:
                prefix = "UID" if command.uid else ""
                if command.name == "FETCH":
                    wire_command = imaplib.FetchCommand(
                        protocol.new_tag(), *args, prefix=prefix, timeout=self.imap.timeout
                    )
                else:
                    wire_command = imaplib.Command(
                        command.name,
                        protocol.new_tag(),
                        *args,
                        prefix=prefix,
                        untagged_resp_name="FETCH" if command.name == "STORE" else command.name,
                        timeout=self.imap.timeout,
                    )
                response = await protocol.execute(wire_command)
        if command.name in ("SELECT", "EXAMINE"):
            protocol.state = imaplib.SELECTED if response.result == "OK" else imaplib.AUTH
        elif command.name == "CLOSE" and response.result == "OK":
            protocol.state = imaplib.AUTH
        return ProtocolStep(
            index=0,
            status="ok" if response.result == "OK" else "error",
            code=response.result,
            responses=[ResponseBlock.from_bytes(line) for line in response.lines],
        )


class LocalProtocolBackend:
    @asynccontextmanager
    async def connect(self, account_name: str, protocol: ProtocolName) -> AsyncIterator[ProtocolSession]:
        initial = resolve_local_account(account_name, fresh_legacy=True)
        require_protocol_access(initial, protocol)
        session = LocalProtocolSession(initial, protocol)
        try:
            await session.open()
            yield session
        finally:
            session.close()
