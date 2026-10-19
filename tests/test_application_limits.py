from __future__ import annotations

from dataclasses import replace

import pytest

from mcp_email_server.application import limits


@pytest.mark.parametrize(
    ("value", "valid"),
    [
        ("aaa", True),
        ("éé", True),
        ("ééa", False),
    ],
)
def test_controlled_string_limit_uses_utf8_bytes_at_limit_minus_one_limit_and_plus_one(
    value: str,
    valid: bool,
) -> None:
    if valid:
        assert (
            limits.validate_controlled_string(
                value,
                field_name="query",
                maximum_bytes=4,
            )
            == value
        )
    else:
        with pytest.raises(ValueError, match="exceeds 4 bytes"):
            limits.validate_controlled_string(
                value,
                field_name="query",
                maximum_bytes=4,
            )


@pytest.mark.parametrize("control", ["\x00", "\x1f", "\x7f"])
def test_controlled_string_rejects_c0_and_del(control: str) -> None:
    with pytest.raises(ValueError, match="control characters"):
        limits.validate_controlled_string(
            f"safe{control}unsafe",
            field_name="query",
            maximum_bytes=100,
        )


@pytest.mark.parametrize(
    ("payload", "valid"),
    [(b"aaa", True), (b"aaaa", True), (b"aaaaa", False)],
)
def test_serialized_result_limit_accepts_limit_minus_one_and_limit_only(
    monkeypatch: pytest.MonkeyPatch,
    payload: bytes,
    valid: bool,
) -> None:
    monkeypatch.setattr(
        limits,
        "APPLICATION_LIMITS",
        replace(limits.APPLICATION_LIMITS, serialized_response_bytes=4),
    )

    if valid:
        limits.validate_serialized_result(payload)
    else:
        with pytest.raises(ValueError, match="serialized result exceeds 4 bytes"):
            limits.validate_serialized_result(payload)


@pytest.mark.parametrize(
    ("decoded", "encoded"),
    [(0, 0), (1, 4), (2, 4), (3, 4), (4, 8), (20 * 1_024 * 1_024, 27_962_028)],
)
def test_base64_encoded_length_matches_padded_rfc4648_output(decoded: int, encoded: int) -> None:
    import base64

    assert limits.base64_encoded_length(decoded) == encoded
    if decoded <= 4:
        assert len(base64.b64encode(b"x" * decoded)) == encoded


def test_base64_encoded_length_rejects_negative_sizes() -> None:
    with pytest.raises(ValueError, match="must not be negative"):
        limits.base64_encoded_length(-1)


def test_inline_attachment_budget_is_bounded_by_existing_attachment_limits() -> None:
    budget = limits.APPLICATION_LIMITS
    assert budget.inline_attachment_total_bytes == 20 * 1_024 * 1_024
    assert budget.inline_attachment_total_bytes <= budget.attachment_bytes
    assert budget.inline_attachment_total_bytes <= budget.total_attachment_bytes
    assert budget.attachment_filename_bytes == 255
    assert budget.attachment_mime_type_bytes == 255


def test_inbound_mcp_message_limits_fit_the_inline_budget_with_bounded_headroom() -> None:
    budget = limits.APPLICATION_LIMITS
    encoded_budget = limits.base64_encoded_length(budget.inline_attachment_total_bytes)
    # Twice the body allows JSON escaping of a maximal body; headers and recipients
    # are a small remainder of the same headroom.
    headroom = 2 * budget.body_bytes + 4 * budget.header_bytes + budget.recipients * budget.address_bytes
    for frame_limit in (budget.mcp_stdio_frame_bytes, budget.mcp_http_request_body_bytes):
        assert encoded_budget + headroom < frame_limit
        assert frame_limit == 32 * 1_024 * 1_024
    # Outbound results keep their independent, unchanged ceiling.
    assert budget.serialized_response_bytes == 8 * 1_024 * 1_024
