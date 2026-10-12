from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager, closing
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from mcp_email_server import app as app_module
from mcp_email_server import managed as managed_module
from mcp_email_server.adapters import protocol as adapter
from mcp_email_server.adapters.authority import LocalAccountResolution
from mcp_email_server.application.protocol import (
    MAX_RECEIVED_BYTES,
    ProtocolCommandService,
    ProtocolFailure,
    ProtocolRequest,
    mailbox_text,
    parse_commands,
    tokenize,
)
from mcp_email_server.config import Settings
from mcp_email_server.managed import ManagedCatalog


def request(*commands, protocol="imap", account_name="test_account"):
    return ProtocolRequest.model_validate({
        "account_name": account_name,
        "protocol": protocol,
        "commands": [{"command": item} if isinstance(item, str) else item for item in commands],
    })


@pytest.mark.parametrize(
    "line",
    [
        "LOGIN user secret",
        "A1 NOOP",
        "STARTTLS",
        "LOGOUT",
        "IDLE",
        "UID IDLE",
        "COMPRESS DEFLATE",
        "SELECT INBOX\r\nNOOP",
        "NOOP\n",
        "APPEND INBOX {12}",
        'SELECT "unfinished',
        "FETCH 1 (BODY[]",
        "NOOP\x00",
        "UID COPY 1",
        'SELECT "one""two"',
    ],
)
def test_framing_and_unsupported_commands_fail_preflight(line):
    with pytest.raises(ProtocolFailure):
        parse_commands(request("CREATE Good", line))


def test_protocol_tokenization_preserves_expressions_and_mailboxes():
    assert tokenize("UID FETCH 1 (BODY.PEEK[HEADER.FIELDS (FROM SUBJECT)] FLAGS)") == (
        "UID",
        "FETCH",
        "1",
        "(BODY.PEEK[HEADER.FIELDS (FROM SUBJECT)] FLAGS)",
    )
    parsed = parse_commands(request('SELECT "项目 A"', 'UID COPY 1:5 "Archive \\"2026\\""'))
    assert parsed[1].uid
    assert mailbox_text(parsed[1].args[1]) == 'Archive "2026"'


@pytest.mark.parametrize(
    "commands",
    [
        [{"command": "NOOP", "data": "secret-message"}],
        [{"command": "APPEND INBOX"}],
        [{"command": "HELP"}, {"command": "NOOP"}],
        [{"command": "APPEND INBOX", "data": "é" * 600_000}],
        [{"command": "SEARCH " + "a" * 40_000}] * 2,
    ],
)
def test_payload_and_aggregate_limits(commands):
    with pytest.raises(ProtocolFailure):
        parse_commands(request(*commands))


@pytest.mark.asyncio
@pytest.mark.parametrize("topic,protocol", [("HELP", "imap"), ("HELP UID COPY", "imap"), ("HELP DATA", "smtp")])
async def test_help_is_local_and_mcp_representations_agree(monkeypatch, topic, protocol):
    runtime = MagicMock(side_effect=AssertionError("HELP must not construct runtime"))
    monkeypatch.setattr(app_module, "get_application_runtime", runtime)
    result = await app_module.mcp.call_tool("email_command", {"commands": [{"command": topic}], "protocol": protocol})
    assert not result.isError
    assert json.loads(result.content[0].text) == result.structuredContent
    assert result.structuredContent["results"][0]["code"] == "HELP"
    runtime.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "arguments",
    [
        {"commands": '[{"command":"HELP"}]'},
        {"commands": [{"command": "HELP", "host": "secret"}]},
        {"commands": [{"command": "HELP"}], "password": "secret"},
        {"commands": [{"command": "HELP"}], "account_name": 12},
    ],
)
async def test_mcp_strict_validation_does_not_echo_input(arguments):
    result = await app_module.mcp.call_tool("email_command", arguments)
    assert result.isError
    assert "secret" not in result.content[0].text


@pytest.mark.asyncio
async def test_json_looking_account_names_are_not_sdk_coerced(monkeypatch):
    service = SimpleNamespace(execute=AsyncMock(return_value=app_module.ProtocolResult(status="completed")))
    monkeypatch.setattr(app_module, "get_application_runtime", lambda: SimpleNamespace(protocol_commands=service))
    await app_module.mcp.call_tool("email_command", {"account_name": "null", "commands": [{"command": "NOOP"}]})
    assert service.execute.call_args.args[0].account_name == "null"


@pytest.fixture
def resolution(email_settings, tmp_path):
    account = email_settings.model_copy(update={"allow_protocol_commands": True})
    settings = Settings.model_construct(
        emails=[account],
        providers=[],
        db_location=str(tmp_path / "index.sqlite3"),
        enable_attachment_content=True,
        allowed_senders=[],
        allowed_recipients=["*"],
    )
    return LocalAccountResolution("legacy", settings, account)


@pytest.mark.parametrize(
    "protocol,account_change,policy_change",
    [
        ("imap", {"allow_protocol_commands": False}, {}),
        ("imap", {"allowed_mutations": ["organize"]}, {}),
        ("imap", {}, {"allowed_senders": ["*"]}),
        ("imap", {}, {"enable_attachment_content": False}),
        ("smtp", {"outgoing": None}, {}),
        ("smtp", {}, {"allowed_recipients": []}),
        ("smtp", {}, {"allowed_recipients": ["*@example.test"]}),
        ("smtp", {"allowed_mutations": []}, {}),
    ],
)
def test_coarse_admission_fails_closed(resolution, protocol, account_change, policy_change):
    changed = replace(
        resolution,
        account=resolution.account.model_copy(update=account_change),
        settings=resolution.settings.model_copy(update=policy_change),
    )
    with pytest.raises(ProtocolFailure):
        adapter.require_protocol_access(changed, protocol)


@pytest.mark.parametrize("change", ["endpoint", "credential", "opt_in", "policy"])
def test_guard_rechecks_connection_identity_and_authority(monkeypatch, resolution, change):
    session = adapter.LocalProtocolSession(resolution, "imap")
    if change == "endpoint":
        account = resolution.account.model_copy(
            update={"incoming": resolution.account.incoming.model_copy(update={"host": "changed.example.test"})}
        )
        changed = replace(resolution, account=account)
    elif change == "credential":
        changed = replace(resolution, authority_revision=("account", 2, 1))
    elif change == "opt_in":
        changed = replace(resolution, account=resolution.account.model_copy(update={"allow_protocol_commands": False}))
    else:
        changed = replace(
            resolution, settings=resolution.settings.model_copy(update={"enable_attachment_content": False})
        )
    monkeypatch.setattr(adapter, "resolve_local_account", lambda *args, **kwargs: changed)
    with pytest.raises(ProtocolFailure):
        session.guard()


@asynccontextmanager
async def imap_server(callback):
    received = []
    connections = []

    async def handle(reader, writer):
        connections.append(writer)
        writer.write(b"* OK synthetic IMAP\r\n")
        await writer.drain()
        try:
            while line := await reader.readline():
                tag, _, command = line.rstrip(b"\r\n").partition(b" ")
                if command == b"CAPABILITY":
                    writer.write(b"* CAPABILITY IMAP4rev1 UIDPLUS MOVE\r\n" + tag + b" OK capabilities\r\n")
                elif command.startswith(b"LOGIN "):
                    writer.write(tag + b" OK authenticated\r\n")
                else:
                    received.append(command)
                    await callback(command, tag, reader, writer)
                await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    try:
        yield server.sockets[0].getsockname()[1], received
    finally:
        server.close()
        await server.wait_closed()
        for writer in connections:
            writer.close()
        await asyncio.sleep(0)


def use_resolution(monkeypatch, resolution, port, protocol="imap"):
    role = "incoming" if protocol == "imap" else "outgoing"
    endpoint = resolution.account.incoming if role == "incoming" else resolution.account.outgoing
    server = endpoint.model_copy(update={"host": "127.0.0.1", "port": port, "use_ssl": False, "start_ssl": False})
    account = resolution.account.model_copy(update={role: server})
    current = replace(
        resolution, account=account, settings=resolution.settings.model_copy(update={"emails": [account]})
    )
    monkeypatch.setattr(adapter, "resolve_local_account", lambda *args, **kwargs: current)
    monkeypatch.setattr(adapter.LocalProtocolSession, "invalidate", AsyncMock())
    return current


@pytest.mark.asyncio
async def test_real_imap_state_literals_unicode_append_and_no_implicit_close(monkeypatch, resolution):
    payload = b"Subject: Test\r\n\r\nhello\xff"
    appended = []

    async def reply(command, tag, reader, writer):
        if command.startswith(b"UID FETCH"):
            writer.write(b"* 1 FETCH (BODY[] {" + str(len(payload)).encode() + b"}\r\n" + payload + b")\r\n")
        if command.startswith(b"APPEND"):
            size = int(command.rsplit(b"{", 1)[1][:-1])
            writer.write(b"+ send literal\r\n")
            await writer.drain()
            appended.append(await reader.readexactly(size + 2))
        writer.write(tag + b" OK [COPYUID 9 1 7] complete\r\n")

    async with imap_server(reply) as (port, wire):
        use_resolution(monkeypatch, resolution, port)
        service = ProtocolCommandService(adapter.LocalProtocolBackend())
        result = await service.execute(
            request(
                'EXAMINE "项目 A"',
                "UID FETCH 1 (BODY.PEEK[])",
                'UID COPY 1 "Bills"',
                {"command": 'APPEND "Drafts" (\\Draft)', "data": "Subject: Test\r\n\r\nHello"},
                'RENAME "Bills" "Archive"',
                "SUBSCRIBE Archive",
            )
        )
        assert result.status == "completed", result
        literal = [block for block in result.results[1].responses if block.kind == "literal"]
        assert len(literal) == 1 and literal[0].encoding == "base64"
        assert appended == [b"Subject: Test\r\n\r\nHello\r\n"]
        assert wire[0] == b'EXAMINE "&mHl27g- A"'
        assert not any(command in (b"CLOSE", b"EXPUNGE", b"LOGOUT") for command in wire)
        independent = await service.execute(request("UID FETCH 1 FLAGS"))
        assert independent.status == "stopped"
        assert independent.results[0].status == "error"


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["reject", "disconnect", "overflow"])
async def test_prior_evidence_and_stop_on_failure(monkeypatch, resolution, failure):
    async def reply(command, tag, reader, writer):
        if command.startswith(b"UID COPY"):
            if failure == "disconnect":
                writer.close()
            elif failure == "overflow":
                writer.write(b"x" * (MAX_RECEIVED_BYTES + 1))
            else:
                writer.write(tag + b" NO [TRYCREATE] missing destination\r\n")
        else:
            writer.write(tag + b" OK selected\r\n")

    async with imap_server(reply) as (port, wire):
        use_resolution(monkeypatch, resolution, port)
        result = await ProtocolCommandService(adapter.LocalProtocolBackend()).execute(
            request("SELECT INBOX", "UID COPY 1 Missing", "NOOP")
        )
        assert result.status == ("stopped" if failure == "reject" else "unknown"), result
        assert [step.status for step in result.results] == [
            "ok",
            "error" if failure == "reject" else "unknown",
            "not_attempted",
        ]
        assert wire == [b'SELECT "INBOX"', b'UID COPY 1 "Missing"']


@pytest.mark.asyncio
async def test_append_revocation_at_continuation_blocks_body(monkeypatch, resolution):
    bodies = []
    current = None

    async def reply(command, tag, reader, writer):
        nonlocal current
        current.account.allow_protocol_commands = False
        writer.write(b"+ continue\r\n")
        await writer.drain()
        bodies.append(await reader.read())

    async with imap_server(reply) as (port, _wire):
        current = use_resolution(monkeypatch, resolution, port)
        result = await ProtocolCommandService(adapter.LocalProtocolBackend()).execute(
            request({"command": "APPEND INBOX", "data": "body-never-sent"})
        )
        assert result.status == "unknown"
        await asyncio.sleep(0.01)
        assert bodies == [b""]


@pytest.mark.asyncio
async def test_full_prevalidation_precedes_backend_access():
    backend = MagicMock()
    result = await ProtocolCommandService(backend).execute(request("CREATE First", "LOGIN forbidden secret"))
    assert result.status == "stopped"
    assert all(step.status == "not_attempted" for step in result.results)
    backend.connect.assert_not_called()


def test_v5_migration_defaults_off_and_preserves_rows(tmp_path):
    import sqlite3

    from tests.test_managed_catalog import _v3_catalog

    catalog = _v3_catalog(tmp_path)
    with closing(sqlite3.connect(catalog.path)) as db:
        db.executescript(managed_module._SCHEMA_V4_ADDITIONS + managed_module._SCHEMA_V5_ADDITIONS)
        db.execute("UPDATE schema_metadata SET version = 5")
        db.execute("UPDATE managed_account SET drafts_mailbox = 'Drafts', allowed_mutations_json = '[\"organize\"]'")
        db.commit()
    details = catalog.show_account("alice")
    assert not details.allow_protocol_commands
    assert details.drafts_mailbox == "Drafts"
    assert details.allowed_mutations == ("organize",)
    assert details.revision == 5
    catalog.update_account("alice", expected_revision=5, allow_protocol_commands=True)
    assert catalog.show_account("alice").allow_protocol_commands
    assert catalog.load_account("alice").allow_protocol_commands
    assert catalog.load_settings().emails[0].allow_protocol_commands
    catalog.update_account("alice", expected_revision=6, full_name="Changed")
    assert catalog.show_account("alice").allow_protocol_commands
    catalog.update_account("alice", expected_revision=7, allow_protocol_commands=False)
    assert not catalog.load_account("alice").allow_protocol_commands
    assert ManagedCatalog(catalog.path).show_account("alice").revision == 8


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scenario", ["accepted", "rcpt_rejected", "data_rejected", "disconnect", "revoke", "malformed", "long_reply"]
)
async def test_real_smtp_data_is_one_step_and_never_replayed(monkeypatch, resolution, scenario):  # noqa: C901 - protocol scenarios
    wire = []
    bodies = []
    current = None
    finished = asyncio.Event()

    async def handle(reader, writer):  # noqa: C901 - synthetic SMTP state machine
        writer.write(b"220 synthetic SMTP\r\n")
        try:
            while line := await reader.readline():
                if line.startswith(b"EHLO"):
                    writer.write(b"250-localhost\r\n250 AUTH PLAIN\r\n")
                elif line.startswith(b"AUTH"):
                    writer.write(b"235 authenticated\r\n")
                else:
                    wire.append(line)
                    if line.startswith(b"RCPT") and scenario == "rcpt_rejected":
                        writer.write(b"550 no recipient\r\n")
                    elif line == b"DATA\r\n":
                        if scenario == "data_rejected":
                            writer.write(b"554 message rejected\r\n")
                            continue
                        if scenario == "revoke":
                            current.account.allow_protocol_commands = False
                        writer.write(b"354 continue\r\n")
                        await writer.drain()
                        body = bytearray()
                        while part := await reader.readline():
                            body.extend(part)
                            if part == b".\r\n":
                                break
                        bodies.append(bytes(body))
                        if scenario == "disconnect":
                            writer.close()
                            break
                        if not part:
                            break
                        if scenario == "malformed":
                            writer.write(b"not a valid SMTP reply\r\n")
                        elif scenario == "long_reply":
                            writer.write(b"250 " + b"x" * 9000 + b"\r\n")
                        else:
                            writer.write(b"250 accepted\r\n")
                    else:
                        writer.write(b"250 accepted\r\n")
                await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()
            finished.set()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    try:
        current = use_resolution(monkeypatch, resolution, server.sockets[0].getsockname()[1], "smtp")
        result = await ProtocolCommandService(adapter.LocalProtocolBackend()).execute(
            request(
                "MAIL FROM:<sender@example.test>",
                "RCPT TO:<recipient@example.test>",
                {"command": "DATA", "data": "Subject: Test\n\n.Hello\n"},
                "NOOP",
                protocol="smtp",
            )
        )
        await asyncio.wait_for(finished.wait(), 2)
        assert result.results[0].status == "ok"
        if scenario == "accepted":
            assert result.status == "completed", result
            assert bodies == [b"Subject: Test\r\n\r\n..Hello\r\n.\r\n"]
            assert wire.count(b"DATA\r\n") == 1
        elif scenario == "rcpt_rejected":
            assert result.status == "stopped", result
            assert result.results[1].code == "550"
            assert result.results[2].status == "not_attempted"
            assert b"DATA\r\n" not in wire
        elif scenario == "data_rejected":
            assert result.status == "stopped", result
            assert result.results[2].code == "554"
            assert not bodies
        else:
            assert result.status == "unknown", result
            assert result.results[2].status == "unknown"
            if scenario == "revoke":
                assert bodies == [b""]
        if scenario != "accepted":
            assert result.results[3].status == "not_attempted"
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["initial", "final"])
async def test_metadata_failure_does_not_rewrite_remote_evidence(monkeypatch, resolution, phase):
    async def reply(command, tag, reader, writer):
        writer.write(tag + b" OK done\r\n")

    async with imap_server(reply) as (port, wire):
        use_resolution(monkeypatch, resolution, port)
        invalidate = AsyncMock(
            side_effect=[RuntimeError("local failure")] if phase == "initial" else [None, RuntimeError("local failure")]
        )
        monkeypatch.setattr(adapter.LocalProtocolSession, "invalidate", invalidate)
        result = await ProtocolCommandService(adapter.LocalProtocolBackend()).execute(request("CREATE Mailbox"))
        if phase == "initial":
            assert not wire
            assert result.status == "stopped"
        else:
            assert result.status == "completed"
            assert result.results[0].code == "OK"
            assert result.warnings


@pytest.mark.asyncio
async def test_imap_receive_cap_precedes_dependency_buffering():
    protocol = adapter.BoundedImapProtocol(asyncio.get_running_loop())
    protocol.transport = MagicMock()
    protocol._handle_responses = MagicMock()
    protocol.data_received(b"x" * (MAX_RECEIVED_BYTES + 1))
    protocol._handle_responses.assert_not_called()
    assert protocol.incomplete_line == b""
    assert protocol.failure is not None
    protocol.transport.close.assert_called_once()


def test_v6_migration_failure_rolls_back_and_retries(tmp_path, monkeypatch):
    import sqlite3

    from tests.test_managed_catalog import _v3_catalog

    catalog = _v3_catalog(tmp_path)
    with closing(sqlite3.connect(catalog.path)) as db:
        db.executescript(managed_module._SCHEMA_V4_ADDITIONS + managed_module._SCHEMA_V5_ADDITIONS)
        db.execute("UPDATE schema_metadata SET version = 5")
        db.commit()
    original = managed_module._execute_schema

    def fail(connection, schema):
        original(connection, schema)
        if schema == managed_module._SCHEMA_V6_ADDITIONS:
            raise RuntimeError("migration failure")

    with monkeypatch.context() as scoped:
        scoped.setattr(managed_module, "_execute_schema", fail)
        with pytest.raises(RuntimeError, match="migration failure"):
            catalog.show_account("alice")
    with closing(sqlite3.connect(catalog.path)) as db:
        assert db.execute("SELECT version FROM schema_metadata").fetchone()[0] == 5
        assert "allow_protocol_commands" not in [row[1] for row in db.execute("PRAGMA table_info(managed_account)")]
    assert not catalog.show_account("alice").allow_protocol_commands


def test_legacy_fresh_policy_and_selected_keyring_role(monkeypatch, resolution, tmp_path):
    from mcp_email_server import keyring_store
    from mcp_email_server.adapters import authority

    config_path = tmp_path / "legacy.toml"
    config_path.write_text(resolution.settings._to_toml(use_keyring=True, credential_storage="keyring"))
    monkeypatch.setitem(authority._UnresolvedLegacySettings.model_config, "toml_file", config_path)
    monkeypatch.setattr(authority, "process_bootstrap", lambda *args: SimpleNamespace(mode="legacy"))
    monkeypatch.setenv("MCP_EMAIL_SERVER_CREDENTIAL_STORAGE", "keyring")
    secrets = MagicMock(return_value="synthetic-secret")
    monkeypatch.setattr(keyring_store, "get_secret", secrets)
    initial = authority.resolve_local_account("test_account", fresh_legacy=True)
    assert initial.account.allow_protocol_commands
    secrets.assert_not_called()
    selected = authority.resolve_local_account("test_account", fresh_legacy=True, roles=("outgoing",))
    secrets.assert_called_once_with("test_account", "outgoing")
    assert selected.account.outgoing.password.get_secret_value() == "synthetic-secret"
    assert selected.account.incoming.password.get_secret_value() == keyring_store.SENTINEL
    assert selected.settings.emails[0].outgoing.password.get_secret_value() == keyring_store.SENTINEL
    config_path.write_text(
        config_path.read_text().replace("allow_protocol_commands = true", "allow_protocol_commands = false")
    )
    session = adapter.LocalProtocolSession(initial, "imap")
    with pytest.raises(ProtocolFailure, match="protocol_disabled"):
        session.guard()
    secrets.assert_called_once()


def test_environment_protocol_opt_in_is_per_account_and_defaults_false(monkeypatch):
    from mcp_email_server.adapters.management import LocalManagementBackend
    from mcp_email_server.config import EmailSettings

    monkeypatch.setenv("MCP_EMAIL_SERVER_EMAIL_ADDRESS", "user@example.test")
    monkeypatch.setenv("MCP_EMAIL_SERVER_PASSWORD", "synthetic")
    monkeypatch.setenv("MCP_EMAIL_SERVER_IMAP_HOST", "imap.example.test")
    monkeypatch.delenv("MCP_EMAIL_SERVER_ALLOW_PROTOCOL_COMMANDS", raising=False)
    assert not EmailSettings.from_env().allow_protocol_commands
    monkeypatch.setenv("MCP_EMAIL_SERVER_ALLOW_PROTOCOL_COMMANDS", "true")
    assert EmailSettings.from_env().allow_protocol_commands
    snapshot = LocalManagementBackend._environment_account_snapshot()
    assert snapshot.allow_protocol_commands


@pytest.mark.asyncio
async def test_preflight_failure_preserves_order_without_runtime(monkeypatch):
    runtime = MagicMock(side_effect=AssertionError("preflight must remain local"))
    monkeypatch.setattr(app_module, "get_application_runtime", runtime)
    result = await app_module.mcp.call_tool(
        "email_command", {"commands": [{"command": "NOOP"}, {"command": "LOGIN user secret"}]}
    )
    assert result.isError
    assert [step["status"] for step in result.structuredContent["results"]] == ["not_attempted"] * 2
    assert "secret" not in result.content[0].text
    runtime.assert_not_called()


def test_account_name_limit_counts_utf8_bytes():
    with pytest.raises(ProtocolFailure, match="account_name"):
        parse_commands(request("NOOP", account_name="é" * 129))


@pytest.mark.asyncio
@pytest.mark.parametrize("sent", [False, True])
async def test_cancellation_preserves_ambiguity_and_closes_before_invalidation(sent):
    events = []
    session = SimpleNamespace(
        sent=sent,
        reset_step=lambda: None,
        execute=AsyncMock(side_effect=asyncio.CancelledError),
        close=lambda: events.append("close"),
        invalidate=AsyncMock(side_effect=lambda: events.append("invalidate")),
    )

    @asynccontextmanager
    async def connect(*args):
        yield session

    result = await ProtocolCommandService(SimpleNamespace(connect=connect)).execute(request("NOOP", "NOOP"))
    assert result.status == ("unknown" if sent else "stopped")
    assert result.results[0].status == ("unknown" if sent else "error")
    assert result.results[1].status == "not_attempted"
    session.execute.assert_awaited_once()
    assert events == ["invalidate", "close", "invalidate"]


@pytest.mark.asyncio
async def test_result_limit_retains_confirmed_code_and_stops(monkeypatch):
    from mcp_email_server.application import protocol

    monkeypatch.setattr(protocol, "MAX_RESULT_BYTES", 100)
    session = SimpleNamespace(
        sent=True,
        reset_step=lambda: None,
        execute=AsyncMock(
            return_value=protocol.ProtocolStep(
                index=0, status="ok", code="OK", responses=[protocol.ResponseBlock(data="x" * 200)]
            )
        ),
        close=MagicMock(),
        invalidate=AsyncMock(),
    )

    @asynccontextmanager
    async def connect(*args):
        yield session

    result = await ProtocolCommandService(SimpleNamespace(connect=connect)).execute(request("NOOP", "NOOP"))
    assert result.status == "stopped"
    assert result.results[0].status == "ok"
    assert result.results[0].code == "OK"
    assert not result.results[0].responses
    assert result.results[1].status == "not_attempted"
    session.execute.assert_awaited_once()


def test_legacy_protocol_opt_in_import_preview_and_apply(monkeypatch, resolution, tmp_path, fake_keyring):
    from typer.testing import CliRunner

    from mcp_email_server import config
    from mcp_email_server.cli import app

    parent = tmp_path / "import"
    parent.mkdir(mode=0o700)
    source = parent / "config.toml"
    source.write_text(resolution.settings._to_toml(use_keyring=False, credential_storage="plaintext"))
    source.chmod(0o600)
    original = source.read_bytes()
    database = parent / "catalog.sqlite3"
    monkeypatch.setenv("MCP_EMAIL_SERVER_CONFIG_PATH", str(source))
    monkeypatch.setattr(config, "CONFIG_PATH", source)
    monkeypatch.setitem(Settings.model_config, "toml_file", source)
    runner = CliRunner()
    initialized = runner.invoke(app, ["config", "init", "--database", str(database)])
    assert initialized.exit_code == 0, initialized.output
    preview = runner.invoke(app, ["config", "import-legacy"])
    assert preview.exit_code == 0, preview.output
    assert "allow_protocol_commands=true" in preview.output
    applied = runner.invoke(app, ["config", "import-legacy", "--apply"], input="IMPORT\n")
    assert applied.exit_code == 0, applied.output
    assert ManagedCatalog(database).show_account("test_account").allow_protocol_commands
    assert source.read_bytes() == original
