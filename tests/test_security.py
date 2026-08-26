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


def test_accepts_confirmation_approved():
    """The signature-inspection guard both call sites (executor,
    Agent Framework approval middleware) share to decide whether a policy's
    check_permission understands confirmation_approved."""
    from agent_gantry.core.security import SecurityPolicy, accepts_confirmation_approved

    assert accepts_confirmation_approved(SecurityPolicy()) is True

    class LegacyPolicy:
        def check_permission(self, tool_name, arguments):
            pass

    assert accepts_confirmation_approved(LegacyPolicy()) is False

    class KwargsPolicy:
        def check_permission(self, tool_name, arguments, **kwargs):
            pass

    assert accepts_confirmation_approved(KwargsPolicy()) is True

    class NotAPolicy:
        pass

    assert accepts_confirmation_approved(NotAPolicy()) is False


def test_confirmation_probe_is_not_counted_against_rate_limit():
    """A call that comes back needing confirmation never executed, and the
    approved replay that follows is the same logical call — so only the
    replay is counted. With a limit of 1, counting the probe too would make
    any confirmation-gated tool permanently unexecutable."""
    import pytest

    from agent_gantry.core.security import (
        ConfirmationRequiredError,
        PermissionDeniedError,
        SecurityPolicy,
    )

    policy = SecurityPolicy(require_confirmation=["delete_*"], max_requests_per_minute=1)

    with pytest.raises(ConfirmationRequiredError):
        policy.check_permission("delete_user", {"id": "1"})

    # Approved replay succeeds: the probe above consumed no quota.
    policy.check_permission("delete_user", {"id": "1"}, confirmation_approved=True)

    # …but the replay itself did, so the next call hits the limit.
    with pytest.raises(PermissionDeniedError, match="Rate limit exceeded"):
        policy.check_permission("other_tool", {})


def test_confirmation_approved_cannot_bypass_the_rate_limit():
    """``confirmation_approved`` reaches check_permission from
    ``ToolCall(require_confirmation=False)`` — a caller-supplied field. It
    must never relax a *denial* check, or any client could lift its own
    rate limit just by setting the flag (PR #381 review)."""
    import pytest

    from agent_gantry.core.security import PermissionDeniedError, SecurityPolicy

    policy = SecurityPolicy(require_confirmation=[], max_requests_per_minute=2)

    policy.check_permission("search", {}, confirmation_approved=True)
    policy.check_permission("search", {}, confirmation_approved=True)

    with pytest.raises(PermissionDeniedError, match="Rate limit exceeded"):
        policy.check_permission("search", {}, confirmation_approved=True)


def test_denied_calls_still_consume_rate_limit_quota():
    """Only the confirmation gate defers accounting — a denial is terminal,
    so it counts, keeping a flood of rejected calls bounded."""
    import pytest

    from agent_gantry.core.security import PermissionDeniedError, SecurityPolicy

    policy = SecurityPolicy(
        require_confirmation=[], allowed_domains=["example.com"], max_requests_per_minute=1
    )

    with pytest.raises(PermissionDeniedError, match="not in allowed_domains"):
        policy.check_permission("fetch", {"url": "https://evil.test/x"})

    with pytest.raises(PermissionDeniedError, match="Rate limit exceeded"):
        policy.check_permission("fetch", {"url": "https://example.com/x"})
