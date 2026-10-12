"""Gateway contracts through the registered MCP handler, not just Python calls."""

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import jsonschema
import pytest
from mcp.types import CallToolRequest, CallToolRequestParams, CallToolResult

from mcp_email_server import app
from mcp_email_server import email_commands as commands
from mcp_email_server.adapters.mutations import ClassicMutationProvider
from mcp_email_server.application.accounts import AccountUnavailableError
from mcp_email_server.application.mutations import (
    BatchMutationOutcome,
    CopyCommand,
    CreateMailboxOutcome,
    MutationProviderError,
    MutationRejectedError,
    TargetMutationOutcome,
)
from mcp_email_server.config import EmailServer
from mcp_email_server.emails.classic import EmailClient
from tests.test_copy_emails import _mock_imap as copy_imap
from tests.test_create_mailbox import _mock_imap as create_imap
from tests.test_mutation_application import _account, _services
from tests.test_stdio_protocol import _server

pytestmark = pytest.mark.asyncio
SENTINEL = "PRIVATE_SENTINEL_DO_NOT_ECHO"


def execute(**changes):
    return {
        "action": "execute",
        "operation": "copy_emails",
        "account_name": "primary",
        "arguments": {"email_ids": ["1", "2"], "destination_mailbox": "Bills"},
        **changes,
    }


async def call(arguments):
    handler = app.mcp._mcp_server.request_handlers[CallToolRequest]
    response = (
        await handler(
            CallToolRequest(
                method="tools/call",
                params=CallToolRequestParams(
                    name="email_command",
                    arguments=arguments,
                ),
            )
        )
    ).root
    assert isinstance(response, CallToolResult)
    assert json.loads(response.content[0].text) == response.structuredContent
    commands.EmailCommandResult.model_validate(response.structuredContent)
    assert response.isError == (response.structuredContent["status"] != "completed")
    return response.structuredContent


@pytest.fixture
def runtime(monkeypatch):
    value = SimpleNamespace(
        mutations=SimpleNamespace(
            copy=SimpleNamespace(
                execute=AsyncMock(
                    return_value=BatchMutationOutcome((
                        TargetMutationOutcome("1", "succeeded"),
                        TargetMutationOutcome("2", "succeeded"),
                    ))
                )
            ),
            create_mailbox=SimpleNamespace(
                execute=AsyncMock(return_value=CreateMailboxOutcome("Bills", "created", False))
            ),
        )
    )
    monkeypatch.setattr(commands, "get_application_runtime", lambda: value)
    return value


async def test_local_discovery_schemas_examples_and_snapshot(monkeypatch):
    runtime = MagicMock(side_effect=AssertionError("Discovery must not obtain runtime or secrets"))
    monkeypatch.setattr(commands, "get_application_runtime", runtime)
    listing = await call({"action": "list"})
    assert [entry["operation"] for entry in listing["data"]["operations"]] == ["copy_emails", "create_mailbox"]
    assert listing["data"]["next_cursor"] is None
    actual = []
    for name, operation in commands.OPERATIONS.items():
        result = await call({"action": "describe", "operation": name})
        data = result["data"]
        jsonschema.Draft202012Validator.check_schema(data["arguments_schema"])
        jsonschema.Draft202012Validator.check_schema(data["data_schema"])
        jsonschema.validate(data["example"]["arguments"], data["arguments_schema"])
        operation.arguments_model.model_validate(data["example"]["arguments"]).command("personal")
        actual.append(data)
    expected = json.loads((Path(__file__).parent / "fixtures/email_command_catalog.json").read_text())
    assert actual == expected
    runtime.assert_not_called()


@pytest.mark.parametrize(
    "raw",
    [
        {},
        {"action": SENTINEL},
        {"action": "list", SENTINEL: SENTINEL},
        {"action": "list", "operation": "copy_emails"},
        {"action": "describe"},
        {"action": "describe", "operation": "copy_emails", "arguments": {}},
        execute(arguments="{}"),
        execute(account_name=[SENTINEL]),
        execute(arguments=None),
        execute(cursor="x"),
        execute(operation="RAW IMAP " + SENTINEL),
    ],
)
async def test_malformed_outer_requests_are_safe_and_pre_effect(monkeypatch, raw, caplog, capsys):
    runtime = MagicMock(side_effect=AssertionError("must not resolve runtime"))
    monkeypatch.setattr(commands, "get_application_runtime", runtime)
    result = await call(raw)
    assert result["status"] == "failed"
    assert result["error"]["code"] == "invalid_request"
    assert len(result["error"]["message"]) > 20
    assert SENTINEL not in json.dumps(result) + caplog.text + str(capsys.readouterr())
    runtime.assert_not_called()


@pytest.mark.parametrize(
    "arguments,field",
    [
        ({"email_ids": ["01"], "destination_mailbox": "Bills"}, "email_ids"),
        ({"email_ids": ["4294967296"], "destination_mailbox": "Bills"}, "email_ids"),
        ({"email_ids": ["\u0661"], "destination_mailbox": "Bills"}, "email_ids"),
        ({"email_ids": ["1", "1"], "destination_mailbox": "Bills"}, "email_ids"),
        ({"email_ids": list(map(str, range(1, 102))), "destination_mailbox": "Bills"}, "email_ids"),
        ({"email_ids": ["1"], "destination_mailbox": "inbox"}, "source_mailbox"),
        ({"email_ids": ["1"], "destination_mailbox": "\r" + SENTINEL}, "destination_mailbox"),
        ({"email_ids": ["1"], "destination_mailbox": "é" * 513}, "destination_mailbox"),
        ({"email_ids": ["1"], "destination_mailbox": "Bills", SENTINEL: SENTINEL}, None),
        ({"email_ids": ["1"], "destination_mailbox": "Bills", "account_name": SENTINEL}, None),
    ],
)
async def test_inner_validation_is_strict_and_actionable(monkeypatch, arguments, field):
    runtime = MagicMock(side_effect=AssertionError("must not resolve runtime"))
    monkeypatch.setattr(commands, "get_application_runtime", runtime)
    result = await call(execute(arguments=arguments))
    assert result["error"]["code"] == "invalid_arguments"
    assert result["error"]["field"] == field
    assert SENTINEL not in json.dumps(result)
    runtime.assert_not_called()


@pytest.mark.parametrize("account", ["null", '["example"]', '{"key":1}'])
async def test_optional_scalar_strings_keep_exact_identity(runtime, account):
    result = await call(execute(account_name=account))
    assert result["status"] == "completed"
    runtime.mutations.copy.execute.assert_awaited_once_with(CopyCommand(account, ("1", "2"), "INBOX", "Bills"))


async def test_null_optional_fields_are_omitted_and_unknown_operation_not_echoed():
    assert (await call({"action": "list", "operation": None, "arguments": None}))["status"] == "completed"
    result = await call({"action": "describe", "operation": "not_registered"})
    assert result["error"]["code"] == "unknown_operation"
    assert "action=list" in result["error"]["message"]
    assert result["operation"] is None


@pytest.mark.parametrize(
    "statuses,expected",
    [
        (["succeeded", "succeeded"], "completed"),
        (["succeeded", "failed"], "partial"),
        (["succeeded", "unknown"], "partial"),
        (["failed", "unknown"], "unknown"),
        (["failed", "failed"], "failed"),
    ],
)
async def test_batch_evidence_status_and_no_replay(runtime, statuses, expected):
    runtime.mutations.copy.execute.return_value = BatchMutationOutcome(
        tuple(TargetMutationOutcome(str(i + 1), status, SENTINEL) for i, status in enumerate(statuses))
    )
    result = await call(execute())
    assert result["status"] == expected
    assert [item["target"] for item in result["data"]["outcomes"]] == ["1", "2"]
    assert SENTINEL not in json.dumps(result)
    if expected != "completed":
        assert commands.NO_REPLAY in result["warnings"]
    runtime.mutations.copy.execute.assert_awaited_once()


async def test_projection_warning_does_not_erase_success(runtime):
    runtime.mutations.copy.execute.return_value = BatchMutationOutcome(
        (
            TargetMutationOutcome("1", "succeeded"),
            TargetMutationOutcome("2", "succeeded"),
        ),
        reconciliation_needed=True,
    )
    result = await call(execute())
    assert result["status"] == "completed"
    assert result["warnings"] and result["reconciliation_needed"]


@pytest.mark.parametrize("status", ["created", "already_exists", "unknown"])
async def test_create_reuses_service_and_validates_data_schema(runtime, status):
    runtime.mutations.create_mailbox.execute.return_value = CreateMailboxOutcome("Bills", status, status == "unknown")
    result = await call(execute(operation="create_mailbox", arguments={"mailbox": "Bills"}))
    assert result["status"] == ("unknown" if status == "unknown" else "completed")
    jsonschema.validate(result["data"], commands.CreateData.model_json_schema())
    runtime.mutations.create_mailbox.execute.assert_awaited_once()


@pytest.mark.parametrize(
    "exception,code,status",
    [
        (AccountUnavailableError(SENTINEL), "account_unavailable", "failed"),
        (PermissionError(SENTINEL), "policy_denied", "failed"),
        (NotImplementedError(SENTINEL), "unsupported_capability", "failed"),
        (MutationRejectedError(SENTINEL), "provider_rejected", "failed"),
        (MutationProviderError(SENTINEL), "provider_failure", "unknown"),
        (RuntimeError(SENTINEL), "execution_unknown", "unknown"),
        (asyncio.CancelledError(SENTINEL), "execution_cancelled", "unknown"),
    ],
)
async def test_service_errors_are_safe_classified_and_never_retried(runtime, exception, code, status, caplog):
    runtime.mutations.copy.execute.side_effect = exception
    result = await call(execute())
    assert result["error"]["code"] == code
    assert result["status"] == status
    assert SENTINEL not in json.dumps(result) + caplog.text
    assert result["reconciliation_needed"] == (status == "unknown")
    if status == "unknown":
        assert result["data"]["destination_mailbox"] == "Bills"
        assert [item["status"] for item in result["data"]["outcomes"]] == ["unknown", "unknown"]
    runtime.mutations.copy.execute.assert_awaited_once()


async def test_gateway_grant_denial_before_provider_open(monkeypatch):
    services, _, factory, _ = _services(account=_account(allowed_mutations=()))
    monkeypatch.setattr(commands, "get_application_runtime", lambda: SimpleNamespace(mutations=services))
    assert (await call(execute()))["error"]["code"] == "policy_denied"
    factory.open.assert_not_called()


async def test_envelope_limits_and_non_json_before_runtime(monkeypatch):
    runtime = MagicMock(side_effect=AssertionError("must not resolve runtime"))
    monkeypatch.setattr(commands, "get_application_runtime", runtime)
    nested = {}
    for _ in range(10):
        nested = {"x": nested}
    for value in (nested, {"x": "x" * 65537}, {"x": float("nan")}, {"x": object()}, {"x": "\ud800"}, {1: "x"}):
        response = await commands.dispatch_email_command(value)
        assert response.structuredContent["error"]["code"] == "invalid_request"
    runtime.assert_not_called()


async def test_pagination_registry_extension_keeps_outer_catalog(monkeypatch):
    before = [tool.model_dump() for tool in await app.mcp.list_tools()]
    definition = commands.OPERATIONS["create_mailbox"]
    monkeypatch.setattr(
        commands,
        "OPERATIONS",
        MappingProxyType({f"test_{index:03}": replace(definition, name=f"test_{index:03}") for index in range(51)}),
    )
    first = await call({"action": "list"})
    assert len(first["data"]["operations"]) == 50
    second = await call({"action": "list", "cursor": first["data"]["next_cursor"]})
    assert len(second["data"]["operations"]) == 1
    assert second["data"]["next_cursor"] is None
    assert (await call({"action": "list", "cursor": SENTINEL}))["error"]["code"] == "invalid_cursor"
    assert [tool.model_dump() for tool in await app.mcp.list_tools()] == before


async def test_full_wire_size_is_checked_and_success_not_rewritten(runtime, monkeypatch):
    checked = []
    original = commands.validate_serialized_result

    def check(raw):
        checked.append(json.loads(raw))
        original(raw)
        if len(checked) == 1:
            raise ValueError("forced encoding ceiling")

    monkeypatch.setattr(commands, "validate_serialized_result", check)
    result = await call(execute())
    assert "content" in checked[0] and "structuredContent" in checked[0]
    assert result["status"] == "completed"
    assert result["error"]["code"] == "result_limit"
    assert commands.NO_REPLAY in result["warnings"]
    runtime.mutations.copy.execute.assert_awaited_once()


async def test_raw_stdio_discovery_and_redacted_errors(tmp_path):
    async with _server(tmp_path) as server:
        await server.initialize()
        cases = [
            ({"action": "list"}, None),
            ({"action": "describe", "operation": "copy_emails"}, None),
            ({"action": "list", SENTINEL: SENTINEL}, "invalid_request"),
            (execute(arguments={"email_ids": [SENTINEL], "destination_mailbox": "Bills"}), "invalid_arguments"),
            (execute(account_name=SENTINEL), "account_unavailable"),
        ]
        for request_id, (arguments, code) in enumerate(cases, 10):
            await server.send({
                "jsonrpc": "2.0",
                "id": request_id,
                "method": "tools/call",
                "params": {"name": "email_command", "arguments": arguments},
            })
            response = (await server.response(request_id))["result"]
            result = response["structuredContent"]
            assert json.loads(response["content"][0]["text"]) == result
            assert (result["error"]["code"] if result["error"] else None) == code
            assert SENTINEL not in json.dumps(response)
        await server.close_stdin()
        assert await server.wait() == 0
        assert SENTINEL.encode() not in server.stderr()


def wire_provider(monkeypatch, imap, fresh_authority=None):
    client = EmailClient(
        EmailServer(user_name="u", password="synthetic", host="example.test", port=993, use_ssl=True),
        sender="test@example.test",
    )
    monkeypatch.setattr(client, "imap_class", lambda *args, **kwargs: imap)
    handler = SimpleNamespace(incoming_client=client, outgoing_client=None)
    provider = ClassicMutationProvider(handler, fresh_authority)
    services, _, _, _ = _services(provider=provider)
    monkeypatch.setattr(commands, "get_application_runtime", lambda: SimpleNamespace(mutations=services))
    return client


async def test_hierarchy_error_is_actionable_pre_effect_through_all_layers(monkeypatch):
    imap = create_imap()
    wire_provider(monkeypatch, imap)
    result = await call(execute(operation="create_mailbox", arguments={"mailbox": "business/"}))
    assert result["status"] == "failed"
    assert result["error"]["code"] == "invalid_arguments"
    assert result["error"]["field"] == "mailbox"
    assert "trailing" in result["error"]["message"]
    assert result["reconciliation_needed"] is False
    imap.create.assert_not_awaited()


@pytest.mark.parametrize("change", ["grant", "sender"])
async def test_mid_batch_authority_change_preserves_prior_success(monkeypatch, change):
    current = _account()
    imap = copy_imap()

    async def effect(*args):
        nonlocal current
        current = _account(allowed_mutations=()) if change == "grant" else _account(allowed_senders=("changed",))
        return "OK", [b"COPY completed"]

    imap.uid.side_effect = effect
    wire_provider(monkeypatch, imap, lambda: current)
    result = await call(execute())
    assert result["status"] == "partial"
    assert [(item["status"], item["detail"]) for item in result["data"]["outcomes"]] == [
        ("succeeded", None),
        ("failed", "mutation-policy"),
    ]
    imap.uid.assert_awaited_once()


async def test_sender_hidden_noop_and_cancellation_through_gateway(monkeypatch):
    imap = copy_imap(uid=asyncio.CancelledError())
    client = wire_provider(monkeypatch, imap)
    monkeypatch.setattr(client, "_blocked_uids", AsyncMock(return_value={"1"}))
    result = await call(execute(arguments={"email_ids": ["1", "2", "3"], "destination_mailbox": "Bills"}))
    assert result["status"] == "partial"
    assert [(item["status"], item["detail"]) for item in result["data"]["outcomes"]] == [
        ("succeeded", None),
        ("unknown", "copy-unknown"),
        ("failed", "not-attempted"),
    ]
    imap.uid.assert_awaited_once()


async def test_concurrent_accounts_do_not_share_request_state(runtime):
    async def effect(command):
        await asyncio.sleep(0)
        return BatchMutationOutcome(tuple(TargetMutationOutcome(uid, "succeeded") for uid in command.email_ids))

    runtime.mutations.copy.execute.side_effect = effect
    results = await asyncio.gather(
        *(
            call(execute(account_name=account, arguments={"email_ids": [uid], "destination_mailbox": mailbox}))
            for account, uid, mailbox in [("one", "1", "One"), ("two", "2", "Two")]
        )
    )
    assert [result["data"]["destination_mailbox"] for result in results] == ["One", "Two"]
    assert {item.args[0].account_name for item in runtime.mutations.copy.execute.await_args_list} == {"one", "two"}
