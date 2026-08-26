from agent_gantry.core.security import validate_description


def test_validate_description_multiline_bypass():
    payload = "test {{\n payload \n}}"
    is_valid, msg = validate_description(payload)
    assert is_valid is False
    assert msg == "Description contains suspicious pattern"


def test_validate_description_valid():
    payload = "This is a normal description"
    is_valid, msg = validate_description(payload)
    assert is_valid is True
    assert msg is None


def test_ssrf_domain_extraction_bypass():
    import pytest

    from agent_gantry.core.security import PermissionDeniedError, SecurityPolicy

    sp = SecurityPolicy(allowed_domains=["example.com"])

    # These URLs parse with a netloc but without a valid hostname
    malicious_urls = [
        "http://@/etc/passwd",
        "http://example.com@",
    ]

    for url in malicious_urls:
        with pytest.raises(PermissionDeniedError, match="not in allowed_domains"):
            sp.check_permission("test_tool", {"url": url})

def test_ssrf_port_bypass():
    import pytest

    from agent_gantry.core.security import PermissionDeniedError, SecurityPolicy

    sp = SecurityPolicy(allowed_domains=["example.com"])

    # This URL parses with a valid hostname "example.com" but invalid port "evil.com".
    # Httpx considers this invalid, but if passed to other systems, might act differently.
    # We should ensure standard malformed ones that lack hostname get rejected.
    malicious_urls = [
        "http://@:evil.com/etc/passwd",
    ]

    for url in malicious_urls:
        with pytest.raises(PermissionDeniedError, match="not in allowed_domains"):
            sp.check_permission("test_tool", {"url": url})


def test_ssrf_invalid_port_with_hostname():
    import pytest

    from agent_gantry.core.security import PermissionDeniedError, SecurityPolicy

    sp = SecurityPolicy(allowed_domains=["example.com"])

    # These URLs parse with a valid-looking hostname ("example.com") but carry a
    # non-numeric port string.  urllib.parse.urlparse("http://example.com:evil.com").port
    # raises ValueError; SecurityPolicy catches this and substitutes "<invalid_domain>",
    # which is not in allowed_domains, so the call must be denied.
    #
    # This prevents SSRF bypasses where a downstream HTTP client (httpx, requests, aiohttp)
    # strips or ignores the malformed port and uses the pre-port substring as the hostname,
    # ultimately connecting to a different host than the one the policy intended to permit.
    malicious_urls = [
        "http://example.com:evil.com/etc/passwd",
        "https://example.com:notaport/admin",
        "http://example.com:@attacker.com/path",
    ]

    for url in malicious_urls:
        with pytest.raises(PermissionDeniedError, match="not in allowed_domains"):
            sp.check_permission("test_tool", {"url": url})


def test_confirmation_approved_skips_pattern_gate_but_not_denials():
    """check_permission(confirmation_approved=True) — the executor's
    ToolCall(require_confirmation=False) approval signal — skips only the
    require_confirmation pattern gate; denial checks (allowed domains) still
    run."""
    import pytest

    from agent_gantry.core.security import (
        ConfirmationRequiredError,
        PermissionDeniedError,
        SecurityPolicy,
    )

    policy = SecurityPolicy(
        require_confirmation=["delete_*"], allowed_domains=["example.com"]
    )

    with pytest.raises(ConfirmationRequiredError):
        policy.check_permission("delete_user", {"id": "1"})

    # Approved: the confirmation gate is skipped …
    policy.check_permission("delete_user", {"id": "1"}, confirmation_approved=True)

    # … but a domain denial still applies even when approved.
    with pytest.raises(PermissionDeniedError):
        policy.check_permission(
            "delete_user",
            {"url": "https://evil.test/x"},
            confirmation_approved=True,
        )
