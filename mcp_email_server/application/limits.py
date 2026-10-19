from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ApplicationLimits:
    """Single immutable owner for direct, provider, and serialized work bounds."""

    mutation_uids: int = 100
    content_email_ids: int = 500
    configured_accounts: int = 1_000
    policy_entries: int = 1_000
    account_name_bytes: int = 256
    account_description_bytes: int = 4 * 1_024
    recipients: int = 100
    address_bytes: int = 1_024
    mailbox_bytes: int = 1_024
    query_bytes: int = 64 * 1_024
    mailboxes: int = 1_000
    mailbox_result_bytes: int = 1 * 1_024 * 1_024
    subject_bytes: int = 64 * 1_024
    body_bytes: int = 1 * 1_024 * 1_024
    aggregate_body_bytes: int = 50 * 1_024 * 1_024
    header_bytes: int = 64 * 1_024
    aggregate_header_bytes: int = 4 * 1_024 * 1_024
    metadata_candidates: int = 10_000
    metadata_snapshot_rows: int = 1_000
    attachments: int = 20
    attachment_path_bytes: int = 4_096
    # Inline attachments name a MIME part, not a path; 255 bytes matches common
    # filesystem name limits on the recipient side.
    attachment_filename_bytes: int = 255
    # RFC 6838 type/subtype: two restricted names of at most 127 characters each.
    attachment_mime_type_bytes: int = 255
    flags: int = 100
    flag_bytes: int = 128
    attachment_bytes: int = 25 * 1_024 * 1_024
    total_attachment_bytes: int = 50 * 1_024 * 1_024
    # Decoded bytes of all base64 attachments carried inside one MCP request.
    inline_attachment_total_bytes: int = 20 * 1_024 * 1_024
    maximum_imap_uid: int = 2**32 - 1
    credential_cleanup_rows: int = 100
    warning_items: int = 100
    error_detail_bytes: int = 4_096
    serialized_response_bytes: int = 8 * 1_024 * 1_024
    # One inbound MCP message must carry the 20 MiB inline-attachment budget as
    # base64 (about 26.7 MiB) plus JSON framing, body, and headers. The same bound
    # applies to stdio frames and to SSE/Streamable HTTP request bodies.
    mcp_stdio_frame_bytes: int = 32 * 1_024 * 1_024
    mcp_http_request_body_bytes: int = 32 * 1_024 * 1_024
    spill_file_bytes: int = 64 * 1_024 * 1_024
    ui_json_body_bytes: int = 1 * 1_024 * 1_024
    provider_timeout_seconds: float = 60.0


APPLICATION_LIMITS = ApplicationLimits()
_CANONICAL_IMAP_UID = re.compile(r"[1-9][0-9]*\Z")


def base64_encoded_length(decoded_bytes: int) -> int:
    """Return the padded RFC 4648 base64 length that encodes ``decoded_bytes``."""

    if decoded_bytes < 0:
        raise ValueError("decoded byte count must not be negative")
    return 4 * ((decoded_bytes + 2) // 3)


def validate_imap_uid(value: object, *, field_name: str = "email_id") -> str:
    """Return one canonical positive decimal ASCII IMAP UID within RFC range."""

    if not isinstance(value, str) or _CANONICAL_IMAP_UID.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a canonical positive decimal IMAP UID")
    if int(value) > APPLICATION_LIMITS.maximum_imap_uid:
        raise ValueError(f"{field_name} exceeds the maximum IMAP UID")
    return value


def contains_c0_or_del(value: str) -> bool:
    """Return whether a string contains an ASCII C0 or DEL control."""

    return any(ord(character) < 0x20 or ord(character) == 0x7F for character in value)


def validate_controlled_string(
    value: object,
    *,
    field_name: str,
    maximum_bytes: int,
    allow_empty: bool = False,
) -> str:
    """Validate a command/query string at the shared application boundary."""

    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")  # noqa: TRY004 - validation contract
    if not allow_empty and not value.strip():
        raise ValueError(f"{field_name} must not be empty")
    if contains_c0_or_del(value):
        raise ValueError(f"{field_name} must not contain control characters")
    if len(value.encode("utf-8")) > maximum_bytes:
        raise ValueError(f"{field_name} exceeds {maximum_bytes} bytes")
    return value


def validate_optional_controlled_string(
    value: object | None,
    *,
    field_name: str,
    maximum_bytes: int,
) -> str | None:
    """Validate an optional controlled string while preserving empty compatibility."""

    if value is None:
        return None
    return validate_controlled_string(
        value,
        field_name=field_name,
        maximum_bytes=maximum_bytes,
        allow_empty=True,
    )


def largest_base64_payload_bytes(overhead_bytes: int) -> int:
    """Return the largest decoded payload whose base64 still fits the serialized-result ceiling."""

    available = APPLICATION_LIMITS.serialized_response_bytes - max(0, overhead_bytes)
    return max(0, available // 4 * 3)


def validate_serialized_result(serialized: bytes) -> None:
    """Reject a canonically encoded application result above the shared ceiling."""

    if len(serialized) > APPLICATION_LIMITS.serialized_response_bytes:
        raise ValueError(f"serialized result exceeds {APPLICATION_LIMITS.serialized_response_bytes} bytes")
