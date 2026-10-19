"""Per-account attachment read policy across legacy TOML/environment and managed SQLite."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest
from pydantic import SecretStr

from mcp_email_server import config as config_module
from mcp_email_server.adapters.authority import resolve_local_account
from mcp_email_server.adapters.protocol import require_protocol_access
from mcp_email_server.adapters.reads import ClassicReadProvider, LocalReadBackend
from mcp_email_server.application.protocol import ProtocolFailure
from mcp_email_server.application.reads import (
    AttachmentContentService,
    AttachmentDownloadService,
    AttachmentPayload,
    DownloadAttachmentCommand,
)
from mcp_email_server.bootstrap import BOOTSTRAP_VERSION
from mcp_email_server.config import EmailServer, Settings, clear_settings_cache
from mcp_email_server.managed import ManagedCatalog
from mcp_email_server.windows_security import ensure_private_parent, harden_private_file

_PAYLOAD = AttachmentPayload("7", "Factura_26338.pdf", "application/pdf", b"%PDF-1.4\n%%EOF\n")


def _private_directory(path: Path) -> Path:
    if os.name == "nt":
        ensure_private_parent(path / "placeholder")
    else:
        path.mkdir(mode=0o700)
        path.chmod(0o700)
    return path


def _private_file(path: Path) -> None:
    if os.name == "nt":
        harden_private_file(path)
    else:
        path.chmod(0o600)


def _legacy_account(name: str, overrides: str = "") -> str:
    return (
        f'[[emails]]\naccount_name = "{name}"\nfull_name = "{name}"\nemail_address = "{name}@dismaur.example"\n'
        f"{overrides}"
        f'[emails.incoming]\nuser_name = "{name}@dismaur.example"\npassword = "{name}-secret"\n'
        'host = "imap.dismaur.example"\nport = 993\nuse_ssl = true\n'
    )


def _use_legacy_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, content: str) -> Path:
    parent = _private_directory(tmp_path / "legacy")
    config_path = parent / "config.toml"
    config_path.write_text('credential_storage = "plaintext"\n' + content)
    _private_file(config_path)
    monkeypatch.setenv("MCP_EMAIL_SERVER_CONFIG_PATH", str(config_path))
    monkeypatch.setattr(config_module, "CONFIG_PATH", config_path)
    monkeypatch.setitem(Settings.model_config, "toml_file", config_path)
    for name in (
        "MCP_EMAIL_SERVER_ENABLE_ATTACHMENT_CONTENT",
        "MCP_EMAIL_SERVER_ENABLE_ATTACHMENT_DOWNLOAD",
        "MCP_EMAIL_SERVER_EMAIL_ADDRESS",
        "MCP_EMAIL_SERVER_ACCOUNT_ENABLE_ATTACHMENT_CONTENT",
        "MCP_EMAIL_SERVER_ACCOUNT_ENABLE_ATTACHMENT_DOWNLOAD",
    ):
        monkeypatch.delenv(name, raising=False)
    clear_settings_cache()
    return config_path


def _content_service() -> AttachmentContentService:
    backend = LocalReadBackend()
    return AttachmentContentService(backend, backend)


def _download_service(artifacts: Mock) -> AttachmentDownloadService:
    backend = LocalReadBackend()
    return AttachmentDownloadService(backend, backend, artifacts)


def _artifacts() -> Mock:
    artifacts = Mock()
    artifacts.preflight = AsyncMock(return_value="/approved/Factura_26338.pdf")
    artifacts.write = AsyncMock(return_value="/approved/Factura_26338.pdf")
    return artifacts


async def _read_content(account_name: str) -> AttachmentPayload:
    with patch.object(ClassicReadProvider, "fetch_attachment", AsyncMock(return_value=_PAYLOAD)) as fetch:
        result = await _content_service().execute(DownloadAttachmentCommand(account_name, "7", "Factura_26338.pdf"))
    fetch.assert_awaited_once()
    return result


async def _assert_content_denied(account_name: str) -> None:
    with (
        patch.object(ClassicReadProvider, "fetch_attachment", AsyncMock(return_value=_PAYLOAD)) as fetch,
        pytest.raises(
            PermissionError, match=rf"^Attachment content transfer is disabled for account '{account_name}'\."
        ),
    ):
        await _content_service().execute(DownloadAttachmentCommand(account_name, "7", "Factura_26338.pdf"))
    fetch.assert_not_awaited()


async def _assert_download_denied(account_name: str) -> None:
    artifacts = _artifacts()
    with (
        patch.object(ClassicReadProvider, "fetch_attachment", AsyncMock(return_value=_PAYLOAD)) as fetch,
        pytest.raises(PermissionError, match=rf"^Attachment download is disabled for account '{account_name}'\."),
    ):
        await _download_service(artifacts).execute(
            DownloadAttachmentCommand(account_name, "7", "Factura_26338.pdf", "/approved/Factura_26338.pdf")
        )
    fetch.assert_not_awaited()
    artifacts.preflight.assert_not_awaited()


# --- Legacy TOML and environment -------------------------------------------------


@pytest.mark.asyncio
async def test_default_legacy_configuration_keeps_attachment_reads_disabled_for_every_account(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _use_legacy_config(monkeypatch, tmp_path, _legacy_account("info") + _legacy_account("contabilidad"))

    for name in ("info", "contabilidad"):
        snapshot = LocalReadBackend().resolve(name)
        assert snapshot.enable_attachment_content is False
        assert snapshot.enable_attachment_download is False
        await _assert_content_denied(name)
        await _assert_download_denied(name)


@pytest.mark.asyncio
async def test_legacy_account_override_enables_content_only_for_contabilidad(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _use_legacy_config(
        monkeypatch,
        tmp_path,
        "enable_attachment_content = false\n"
        + _legacy_account("info")
        + _legacy_account("contabilidad", "enable_attachment_content = true\n"),
    )

    assert await _read_content("contabilidad") == _PAYLOAD
    await _assert_content_denied("info")
    # Content transfer does not imply filesystem downloads for the same account.
    await _assert_download_denied("contabilidad")


@pytest.mark.asyncio
async def test_legacy_account_false_override_wins_over_an_enabled_global_policy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _use_legacy_config(
        monkeypatch,
        tmp_path,
        "enable_attachment_content = true\nenable_attachment_download = true\n"
        + _legacy_account("info", "enable_attachment_content = false\nenable_attachment_download = false\n")
        + _legacy_account("contabilidad"),
    )

    await _assert_content_denied("info")
    await _assert_download_denied("info")
    assert await _read_content("contabilidad") == _PAYLOAD
    artifacts = _artifacts()
    with patch.object(ClassicReadProvider, "fetch_attachment", AsyncMock(return_value=_PAYLOAD)):
        response = await _download_service(artifacts).execute(
            DownloadAttachmentCommand("contabilidad", "7", "Factura_26338.pdf", "/approved/Factura_26338.pdf")
        )
    assert response.size == len(_PAYLOAD.content)


@pytest.mark.asyncio
async def test_legacy_global_environment_override_still_yields_to_an_explicit_account_value(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _use_legacy_config(
        monkeypatch,
        tmp_path,
        _legacy_account("info", "enable_attachment_content = false\n") + _legacy_account("contabilidad"),
    )
    monkeypatch.setenv("MCP_EMAIL_SERVER_ENABLE_ATTACHMENT_CONTENT", "true")
    clear_settings_cache()

    assert LocalReadBackend().resolve("contabilidad").enable_attachment_content is True
    assert LocalReadBackend().resolve("info").enable_attachment_content is False


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, None), ("true", True), ("1", True), ("false", False), ("0", False), ("unexpected", False)],
)
def test_environment_account_accepts_tri_state_attachment_overrides(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, value: str | None, expected: bool | None
) -> None:
    _use_legacy_config(monkeypatch, tmp_path, "")
    monkeypatch.setenv("MCP_EMAIL_SERVER_ACCOUNT_NAME", "contabilidad")
    monkeypatch.setenv("MCP_EMAIL_SERVER_EMAIL_ADDRESS", "contabilidad@dismaur.example")
    monkeypatch.setenv("MCP_EMAIL_SERVER_PASSWORD", "environment-secret")
    monkeypatch.setenv("MCP_EMAIL_SERVER_IMAP_HOST", "imap.dismaur.example")
    for name in ("CONTENT", "DOWNLOAD"):
        variable = f"MCP_EMAIL_SERVER_ACCOUNT_ENABLE_ATTACHMENT_{name}"
        if value is None:
            monkeypatch.delenv(variable, raising=False)
        else:
            monkeypatch.setenv(variable, value)

    account = Settings().get_account("contabilidad")

    assert account is not None
    assert account.enable_attachment_content is expected  # type: ignore[union-attr]
    assert account.enable_attachment_download is expected  # type: ignore[union-attr]


def test_protocol_admission_uses_the_effective_account_content_policy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _use_legacy_config(
        monkeypatch,
        tmp_path,
        "enable_attachment_content = false\n"
        + _legacy_account("info", "allow_protocol_commands = true\n")
        + _legacy_account("contabilidad", "allow_protocol_commands = true\nenable_attachment_content = true\n"),
    )

    require_protocol_access(resolve_local_account("contabilidad"), "imap")
    with pytest.raises(ProtocolFailure, match="attachment content enabled"):
        require_protocol_access(resolve_local_account("info"), "imap")


# --- Managed SQLite catalog ------------------------------------------------------


def _managed_catalog(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, global_content: bool) -> ManagedCatalog:
    parent = _private_directory(tmp_path / "managed")
    database = parent / "catalog.sqlite3"
    catalog = ManagedCatalog.initialize(database)
    for name in ("info", "contabilidad"):
        catalog.add_account(
            name=name,
            full_name=name,
            email_address=f"{name}@dismaur.example",
            incoming=EmailServer(
                user_name=f"{name}@dismaur.example",
                password=SecretStr("unused"),
                host="imap.dismaur.example",
                port=993,
            ),
            outgoing=None,
        )
        catalog.set_secret(name, "incoming", f"{name}-secret")
    policy = catalog.policy()
    catalog.update_policy(
        expected_revision=policy.revision,
        enable_attachment_download=False,
        enable_attachment_content=global_content,
        allowed_recipients=policy.allowed_recipients,
        allowed_senders=policy.allowed_senders,
        report_blocked_mutations=policy.report_blocked_mutations,
    )
    config_path = parent / "config.toml"
    config_path.write_text(
        f'bootstrap_version = {BOOTSTRAP_VERSION}\nmode = "managed"\ndb_location = "{database.as_posix()}"\n'
        'credential_storage = "plaintext"\n'
    )
    _private_file(config_path)
    monkeypatch.setenv("MCP_EMAIL_SERVER_CONFIG_PATH", str(config_path))
    monkeypatch.setattr(config_module, "CONFIG_PATH", config_path)
    clear_settings_cache()
    return catalog


def _set_content_override(catalog: ManagedCatalog, name: str, value: bool | None) -> None:
    _account_id, revision = catalog.account_revision(name)
    catalog.update_account(
        name, expected_revision=revision, enable_attachment_content=value, update_attachment_content=True
    )


@pytest.mark.asyncio
async def test_managed_defaults_keep_attachment_reads_disabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake_keyring
) -> None:
    catalog = _managed_catalog(monkeypatch, tmp_path, global_content=False)

    for name in ("info", "contabilidad"):
        assert catalog.show_account(name).enable_attachment_content is None
        await _assert_content_denied(name)
        await _assert_download_denied(name)


@pytest.mark.asyncio
async def test_managed_account_override_enables_content_only_for_contabilidad(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake_keyring
) -> None:
    catalog = _managed_catalog(monkeypatch, tmp_path, global_content=False)
    _set_content_override(catalog, "contabilidad", True)

    assert await _read_content("contabilidad") == _PAYLOAD
    await _assert_content_denied("info")

    # Resetting to inherit restores the global value without touching other fields.
    _set_content_override(catalog, "contabilidad", None)
    assert catalog.show_account("contabilidad").enable_attachment_content is None
    await _assert_content_denied("contabilidad")


@pytest.mark.asyncio
async def test_managed_false_override_wins_over_enabled_global_content(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake_keyring
) -> None:
    catalog = _managed_catalog(monkeypatch, tmp_path, global_content=True)
    _set_content_override(catalog, "info", False)

    await _assert_content_denied("info")
    assert await _read_content("contabilidad") == _PAYLOAD


@pytest.mark.asyncio
async def test_managed_override_revoked_after_fetch_is_denied_by_the_current_recheck(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake_keyring
) -> None:
    catalog = _managed_catalog(monkeypatch, tmp_path, global_content=False)
    _set_content_override(catalog, "contabilidad", True)

    async def revoke_during_fetch(*_args, **_kwargs) -> AttachmentPayload:
        _set_content_override(catalog, "contabilidad", False)
        return _PAYLOAD

    with (
        patch.object(ClassicReadProvider, "fetch_attachment", side_effect=revoke_during_fetch),
        pytest.raises(PermissionError, match="disabled for account 'contabilidad'"),
    ):
        await _content_service().execute(DownloadAttachmentCommand("contabilidad", "7", "Factura_26338.pdf"))
