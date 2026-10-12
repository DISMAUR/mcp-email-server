"""Bounded, local discovery and typed adapters for long-tail email workflows.

The registry contains no accounts, credentials, providers, or mutable sessions.
Application services remain the sole owners of authority and provider effects.
"""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from types import MappingProxyType
from typing import Annotated, Any, Literal

from mcp.types import CallToolResult, TextContent
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, ValidationError, field_validator

from mcp_email_server.application.accounts import AccountUnavailableError
from mcp_email_server.application.limits import (
    APPLICATION_LIMITS as LIMITS,
)
from mcp_email_server.application.limits import (
    validate_controlled_string,
    validate_imap_uid,
    validate_serialized_result,
)
from mcp_email_server.application.mutations import (
    CopyCommand,
    CreateMailboxCommand,
    MailboxHierarchyError,
    MutationProviderError,
    MutationRejectedError,
)
from mcp_email_server.runtime import get_application_runtime

Action = Literal["list", "describe", "execute"]
Status = Literal["completed", "partial", "failed", "unknown"]
OperationName = Annotated[str, Field(max_length=LIMITS.command_operation_characters, pattern=r"^[a-z][a-z0-9_]*$")]
AccountName = Annotated[str, Field(min_length=1, max_length=LIMITS.account_name_bytes)]
Mailbox = Annotated[str, Field(min_length=1, max_length=LIMITS.mailbox_bytes)]
Uid = Annotated[
    str,
    Field(max_length=10, pattern=r"^[1-9][0-9]*$"),
    AfterValidator(validate_imap_uid),
]
NO_REPLAY = "Do not replay the whole request or unknown targets automatically; inspect the destination first."


class StrictModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", validate_default=True)


class CommandRequest(StrictModel):
    action: Action
    operation: OperationName | None = None
    account_name: AccountName | None = None
    arguments: dict[str, Any] | None = None
    cursor: Annotated[str, Field(max_length=LIMITS.command_cursor_bytes)] | None = None

    @field_validator("account_name")
    @classmethod
    def validate_account(cls, value: str | None) -> str | None:
        if value is not None:
            validate_controlled_string(value, field_name="account_name", maximum_bytes=LIMITS.account_name_bytes)
        return value


class CommandError(StrictModel):
    code: str
    message: Annotated[str, Field(max_length=LIMITS.error_detail_bytes)]
    field: str | None = None


class EmailCommandResult(StrictModel):
    action: Action | None = None
    operation: str | None = None
    status: Status
    data: dict[str, Any] | None = None
    error: CommandError | None = None
    warnings: list[str] = Field(default_factory=list, max_length=LIMITS.warning_items)
    reconciliation_needed: bool = False


class CopyArguments(StrictModel):
    email_ids: list[Uid] = Field(min_length=1, max_length=LIMITS.mutation_uids)
    destination_mailbox: Mailbox
    source_mailbox: Mailbox = "INBOX"

    @field_validator("email_ids")
    @classmethod
    def unique_uids(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("duplicate UIDs")
        return value

    @field_validator("source_mailbox", "destination_mailbox")
    @classmethod
    def mailbox_bytes(cls, value: str) -> str:
        return validate_controlled_string(value, field_name="mailbox", maximum_bytes=LIMITS.mailbox_bytes)

    @field_validator("source_mailbox")
    @classmethod
    def distinct_mailboxes(cls, value: str, info) -> str:
        destination = info.data.get("destination_mailbox")
        if destination == value or (value.casefold() == "inbox" and str(destination).casefold() == "inbox"):
            raise ValueError("source and destination must differ")
        return value

    def command(self, account: str) -> CopyCommand:
        command = CopyCommand(account, tuple(self.email_ids), self.source_mailbox, self.destination_mailbox)
        command.validate()
        return command


class CreateArguments(StrictModel):
    mailbox: Mailbox

    @field_validator("mailbox")
    @classmethod
    def mailbox_name(cls, value: str) -> str:
        CreateMailboxCommand("validation", value).validate()
        return value

    def command(self, account: str) -> CreateMailboxCommand:
        command = CreateMailboxCommand(account, self.mailbox)
        command.validate()
        return command


class TargetResult(StrictModel):
    target: Uid
    status: Literal["succeeded", "failed", "unknown"]
    detail: str | None = None


class CopyData(StrictModel):
    source_mailbox: Mailbox
    destination_mailbox: Mailbox
    outcomes: list[TargetResult] = Field(min_length=1, max_length=LIMITS.mutation_uids)


class CreateData(StrictModel):
    mailbox: Mailbox
    status: Literal["created", "already_exists", "unknown"]


class IndexEntry(StrictModel):
    operation: str
    summary: str


class IndexData(StrictModel):
    operations: list[IndexEntry] = Field(max_length=LIMITS.command_page_items)
    next_cursor: str | None = None


class DescriptorData(StrictModel):
    operation: str
    summary: str
    arguments_schema: dict[str, Any]
    data_schema: dict[str, Any]
    example: dict[str, Any]
    required_grants: list[str]
    effect_notes: str
    reconciliation: str


@dataclass(frozen=True)
class Operation:
    name: str
    summary: str
    arguments_model: type[CopyArguments] | type[CreateArguments]
    data_model: type[CopyData] | type[CreateData]
    example_arguments: str  # Immutable JSON, decoded afresh for each descriptor.
    effect_notes: str
    handler: Callable[[CopyCommand | CreateMailboxCommand], Awaitable[EmailCommandResult]]

    def describe(self) -> DescriptorData:
        return DescriptorData(
            operation=self.name,
            summary=self.summary,
            arguments_schema=self.arguments_model.model_json_schema(),
            data_schema=self.data_model.model_json_schema(),
            example={
                "action": "execute",
                "operation": self.name,
                "account_name": "personal",
                "arguments": json.loads(self.example_arguments),
            },
            required_grants=["organize"],
            effect_notes=self.effect_notes,
            reconciliation=NO_REPLAY,
        )


_PUBLIC_COPY_DETAILS = frozenset({
    "sender-policy",
    "mutation-policy",
    "not-found",
    "not-attempted",
    "copy-rejected",
    "copy-unknown",
    "provider-timeout",
})


def _batch_status(statuses: list[str]) -> Status:
    if all(status == "succeeded" for status in statuses):
        return "completed"
    if "succeeded" in statuses:
        return "partial"
    return "unknown" if "unknown" in statuses else "failed"


async def _copy(command: CopyCommand | CreateMailboxCommand) -> EmailCommandResult:
    if not isinstance(command, CopyCommand):
        raise TypeError("COPY requires CopyCommand")
    outcome = await get_application_runtime().mutations.copy.execute(command)
    # The service owns effect evidence; never expose unrecognized provider detail.
    data = CopyData(
        source_mailbox=command.source_mailbox,
        destination_mailbox=command.destination_mailbox,
        outcomes=[
            TargetResult(
                target=item.target,
                status=item.status,
                detail=item.detail if item.detail in _PUBLIC_COPY_DETAILS else None,
            )
            for item in outcome.outcomes
        ],
    )
    if [item.target for item in data.outcomes] != list(command.email_ids):
        raise ValueError("Provider outcomes do not match requested targets")
    status = _batch_status([item.status for item in data.outcomes])
    warnings = [NO_REPLAY] if status != "completed" else []
    if outcome.reconciliation_needed:
        warnings.append("Reconcile the destination and refresh metadata; projection warnings do not undo successes.")
    return EmailCommandResult(
        action="execute",
        operation="copy_emails",
        status=status,
        data=data.model_dump(mode="json"),
        warnings=warnings,
        reconciliation_needed=outcome.reconciliation_needed,
    )


async def _create(command: CopyCommand | CreateMailboxCommand) -> EmailCommandResult:
    if not isinstance(command, CreateMailboxCommand):
        raise TypeError("CREATE requires CreateMailboxCommand")
    outcome = await get_application_runtime().mutations.create_mailbox.execute(command)
    data = CreateData(mailbox=command.mailbox, status=outcome.status)
    unknown = outcome.status == "unknown"
    return EmailCommandResult(
        action="execute",
        operation="create_mailbox",
        status="unknown" if unknown else "completed",
        data=data.model_dump(mode="json"),
        warnings=[NO_REPLAY] if unknown else [],
        reconciliation_needed=outcome.reconciliation_needed,
    )


OPERATIONS = MappingProxyType({
    operation.name: operation
    for operation in (
        Operation(
            "copy_emails",
            "Copy selected messages within one account, retaining the originals.",
            CopyArguments,
            CopyData,
            '{"email_ids":["123","456"],"source_mailbox":"INBOX","destination_mailbox":"Bills"}',
            "Requires IMAP and organize. UID COPY only: no MOVE, STORE, EXPUNGE, or implicit CREATE. "
            "The destination must exist and differ from the source (INBOX is case-insensitive). "
            "Sender policy applies; hidden blocked messages remain successful no-ops unless reporting is enabled. "
            "Non-idempotent: repeating can create duplicates. Source UID preflight is not atomic with COPY; "
            "no destination UID, deduplication, or exactly-once guarantee.",
            _copy,
        ),
        Operation(
            "create_mailbox",
            "Create an exact mailbox name, or report that it already exists.",
            CreateArguments,
            CreateData,
            '{"mailbox":"Bills"}',
            "Requires IMAP and organize. Uses the existing mailbox creation service. "
            "No LIST wildcards or trailing server hierarchy delimiter. Returns created, already_exists, or unknown. "
            "No messages are moved or copied.",
            _create,
        ),
    )
})


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")


def _container_size(value: dict | list, depth: int) -> int:
    size = 2
    items = value.items() if isinstance(value, dict) else ((None, item) for item in value)
    for key, item in items:
        if isinstance(value, dict):
            if not isinstance(key, str):
                raise TypeError("JSON object keys must be strings")
            size += len(key.encode("utf-8")) + 3
        size += _check_json(item, depth + 1) + 1
        if size > LIMITS.command_envelope_bytes:
            raise ValueError("JSON size limit")
    return size


def _check_json(value: Any, depth: int = 0) -> int:
    """Bound depth and aggregate size before recursive JSON encoding or validation."""
    if depth > LIMITS.command_json_depth:
        raise ValueError("JSON nesting limit")
    if isinstance(value, str):
        size = len(value.encode("utf-8"))
    elif value is None or isinstance(value, bool):
        size = 4
    elif isinstance(value, int):
        if value.bit_length() > 64:
            raise ValueError("JSON integer limit")
        size = 20
    elif isinstance(value, float) and math.isfinite(value):
        size = 24
    elif isinstance(value, (dict, list)):
        size = _container_size(value, depth)
    else:
        raise ValueError("Only finite JSON values are accepted")
    if size > LIMITS.command_envelope_bytes:
        raise ValueError("JSON size limit")
    return size


_FIELD_HELP = {
    "action": "Use action list, describe, or execute.",
    "operation": "Use a lowercase operation identifier (at most 64 ASCII characters); call list for supported names.",
    "account_name": "Use an exact account_name from list_available_accounts (1-256 UTF-8 bytes, no controls).",
    "arguments": "Supply arguments as a JSON object, not a JSON-encoded string; call describe for its schema.",
    "cursor": "Use only next_cursor returned by list, or omit cursor to start again.",
    "email_ids": "Use 1-100 unique canonical decimal UID strings in the range 1-4294967295, without leading zeros.",
    "source_mailbox": "Use a nonempty mailbox name (at most 1024 UTF-8 bytes, no controls), different from destination.",
    "destination_mailbox": "Use an existing, distinct mailbox name (at most 1024 UTF-8 bytes, no controls).",
    "mailbox": "Use an exact nonempty mailbox name (at most 1024 UTF-8 bytes), without controls or * / % wildcards.",
}


def _validation_error(exc: ValidationError, code: str, request: CommandRequest | None = None) -> EmailCommandResult:
    # Neither error messages, supplied values, nor arbitrary field-name text are safe.
    error = exc.errors(include_url=False, include_context=False, include_input=False)[0]
    location = error["loc"]
    field = location[0] if location and location[0] in _FIELD_HELP else None
    if error["type"] == "extra_forbidden":
        message = "Remove unrecognized fields; use only fields in the declared schema."
        field = None
    else:
        message = _FIELD_HELP[field] if field else "Use the declared JSON object schema; call describe for arguments."
    return _failure(code, message, request, field=field)


def _failure(
    code: str,
    message: str,
    request: CommandRequest | None = None,
    *,
    field: str | None = None,
    unknown: bool = False,
) -> EmailCommandResult:
    return EmailCommandResult(
        action=request.action if request else None,
        operation=request.operation if request and request.operation in OPERATIONS else None,
        status="unknown" if unknown else "failed",
        error=CommandError(code=code, message=message, field=field),
        warnings=[NO_REPLAY] if unknown else [],
        reconciliation_needed=unknown,
    )


def _action_error(request: CommandRequest) -> EmailCommandResult | None:
    required = {"list": (), "describe": ("operation",), "execute": ("operation", "account_name", "arguments")}
    allowed = {
        "list": {"action", "cursor"},
        "describe": {"action", "operation"},
        "execute": {"action", "operation", "account_name", "arguments"},
    }
    values = request.model_dump()
    for field in required[request.action]:
        if values[field] is None:
            return _failure(
                "invalid_request", f"This action requires {field}. {_FIELD_HELP[field]}", request, field=field
            )
    for field, value in values.items():
        if value is not None and field not in allowed[request.action]:
            return _failure(
                "invalid_request", f"Omit {field} for this action (or set it to null).", request, field=field
            )
    return None


def _discovery(request: CommandRequest) -> EmailCommandResult:
    if request.action == "describe":
        data = OPERATIONS[request.operation].describe()  # type: ignore[index]
    else:
        names = sorted(OPERATIONS)
        cursors = {
            f"page-{offset}": offset
            for offset in range(LIMITS.command_page_items, len(names), LIMITS.command_page_items)
        }
        if request.cursor is not None and request.cursor not in cursors:
            return _failure("invalid_cursor", _FIELD_HELP["cursor"], request, field="cursor")
        start = cursors[request.cursor] if request.cursor else 0
        end = start + LIMITS.command_page_items
        data = IndexData(
            operations=[IndexEntry(operation=name, summary=OPERATIONS[name].summary) for name in names[start:end]],
            next_cursor=f"page-{end}" if end < len(names) else None,
        )
    payload = data.model_dump(mode="json")
    if len(_json_bytes(payload)) > LIMITS.command_discovery_bytes:
        return _failure("result_limit", "Discovery exceeds the supported size; report this server defect.", request)
    return EmailCommandResult(action=request.action, operation=request.operation, status="completed", data=payload)


async def _execute(request: CommandRequest) -> EmailCommandResult:
    operation = OPERATIONS[request.operation]  # type: ignore[index]
    try:
        arguments = operation.arguments_model.model_validate(request.arguments)
        command = arguments.command(request.account_name)  # type: ignore[arg-type]
    except ValidationError as exc:
        return _validation_error(exc, "invalid_arguments", request)
    except ValueError:
        return _failure(
            "invalid_arguments",
            "Source and destination must differ, including INBOX casing; "
            "call describe and check UID and mailbox constraints.",
            request,
        )
    result = await _invoke(request, operation, command)
    if result.status == "unknown" and result.data is None:
        # With no service evidence, all requested targets remain ambiguous.
        # Retain validated destination context so reconciliation is actionable.
        if isinstance(command, CopyCommand):
            data = CopyData(
                source_mailbox=command.source_mailbox,
                destination_mailbox=command.destination_mailbox,
                outcomes=[TargetResult(target=uid, status="unknown") for uid in command.email_ids],
            )
        else:
            data = CreateData(mailbox=command.mailbox, status="unknown")
        result.data = data.model_dump(mode="json")
    return result


async def _invoke(
    request: CommandRequest,
    operation: Operation,
    command: CopyCommand | CreateMailboxCommand,
) -> EmailCommandResult:
    try:
        return await operation.handler(command)
    except MailboxHierarchyError:
        return _failure(
            "invalid_arguments",
            "Remove the trailing server hierarchy delimiter from mailbox; CREATE was not attempted.",
            request,
            field="mailbox",
        )
    except AccountUnavailableError:
        return _failure(
            "account_unavailable",
            "Account is missing or disabled. Use list_available_accounts and select "
            "an exact account name; ask the user to configure accounts in CLI/UI, never in chat.",
            request,
        )
    except NotImplementedError:
        return _failure(
            "unsupported_capability",
            "This operation requires an IMAP email account. "
            "Use list_available_accounts and select an account with can_receive=true.",
            request,
        )
    except PermissionError:
        return _failure(
            "policy_denied",
            "Current account policy denies this operation. Ask the user to review the "
            "organize grant and sender policy in CLI/UI; do not bypass policy.",
            request,
        )
    except MutationRejectedError:
        return _failure(
            "provider_rejected",
            "The server rejected mailbox creation. Check the mailbox name and "
            "server permissions; no mailbox was created by this call.",
            request,
        )
    except MutationProviderError:
        return _failure(
            "provider_failure",
            "The provider did not return usable effect evidence. Check account/provider "
            "health in CLI/UI and reconcile the destination before considering another call.",
            request,
            unknown=True,
        )
    except asyncio.CancelledError:
        return _failure(
            "execution_cancelled", "Execution was interrupted; effects may have started.", request, unknown=True
        )
    except Exception:
        return _failure(
            "execution_unknown",
            "Execution did not return usable evidence. Inspect the destination and "
            "account setup in CLI/UI; report persistent failures without sharing credentials.",
            request,
            unknown=True,
        )


def render_result(result: EmailCommandResult) -> CallToolResult:
    """Validate once and derive both wire representations from that same envelope."""
    payload = EmailCommandResult.model_validate(result.model_dump()).model_dump(mode="json")
    response = CallToolResult(
        content=[TextContent(type="text", text=_json_bytes(payload).decode("utf-8"))],
        structuredContent=payload,
        isError=result.status != "completed",
    )
    validate_serialized_result(_json_bytes(response.model_dump(mode="json", exclude_none=True)))
    return response


def _validate_json_envelope(raw: Any) -> None:
    _check_json(raw)
    if len(_json_bytes(raw)) > LIMITS.command_envelope_bytes:
        raise ValueError("Command envelope limit")


async def dispatch_email_command(raw: Any) -> CallToolResult:
    """Scoped raw boundary; deliberately bypass FastMCP's optional-string JSON coercion."""
    try:
        _validate_json_envelope(raw)
    except (ValueError, TypeError, UnicodeError, RecursionError):
        return render_result(
            _failure(
                "invalid_request",
                "Supply finite JSON within 64 KiB and nesting depth 8; use a smaller command and no non-JSON values.",
            )
        )
    try:
        request = CommandRequest.model_validate(raw)
    except ValidationError as exc:
        return render_result(_validation_error(exc, "invalid_request"))
    result = _action_error(request)
    if result is None and request.operation is not None and request.operation not in OPERATIONS:
        result = _failure(
            "unknown_operation",
            "Unsupported operation. Call email_command with action=list, then "
            "describe a supported operation; no raw protocol commands are accepted.",
            request,
            field="operation",
        )
    if result is None:
        result = await _execute(request) if request.action == "execute" else _discovery(request)
    try:
        return render_result(result)
    except (ValueError, TypeError, UnicodeError):
        # Do not turn acknowledged effects into a pre-effect failure when encoding fails.
        fallback = EmailCommandResult(
            action=result.action,
            operation=result.operation,
            status=result.status if request.action == "execute" else "failed",
            error=CommandError(
                code="result_limit",
                message="Result could not be represented within server limits. "
                "Known status is retained; inspect the destination before any further mutation.",
            ),
            warnings=[NO_REPLAY],
            reconciliation_needed=request.action == "execute",
        )
        return render_result(fallback)


async def email_command(
    action: Action,
    operation: OperationName | None = None,
    account_name: AccountName | None = None,
    arguments: dict[str, Any] | None = None,
    cursor: Annotated[str, Field(max_length=LIMITS.command_cursor_bytes)] | None = None,
) -> Annotated[CallToolResult, EmailCommandResult]:
    return await dispatch_email_command({
        "action": action,
        "operation": operation,
        "account_name": account_name,
        "arguments": arguments,
        "cursor": cursor,
    })


# Fail at construction, not after account/provider effects, if help grows beyond its contract.
for _operation in OPERATIONS.values():
    if len(_json_bytes(_operation.describe().model_dump(mode="json"))) > LIMITS.command_discovery_bytes:
        raise ValueError("Operation descriptor exceeds the discovery limit")
