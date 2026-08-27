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


def test_accepts_keyword_generalizes_the_signature_check():
    from agent_gantry.core.security import SecurityPolicy, accepts_keyword

    policy = SecurityPolicy()
    assert accepts_keyword(policy, "confirmation_approved") is True
    assert accepts_keyword(policy, "pending_confirmation") is True
    assert accepts_keyword(policy, "not_a_real_keyword") is False

    class LegacyPolicy:
        def check_permission(self, tool_name, arguments):
            pass

    assert accepts_keyword(LegacyPolicy(), "pending_confirmation") is False


def test_pending_confirmation_defers_accounting_without_relaxing_checks():
    """A gate the *executor* owns (ToolDefinition.requires_confirmation) is
    invisible to the policy, so the executor tells it. Every check still
    runs; only the recording is deferred to the replay."""
    import pytest

    from agent_gantry.core.security import PermissionDeniedError, SecurityPolicy

    policy = SecurityPolicy(
        require_confirmation=[], allowed_domains=["example.com"], max_requests_per_minute=2
    )

    # Probe: passes every check, but is not recorded — so several in a row
    # never exhaust a budget of 2.
    for _ in range(4):
        policy.check_permission("risky_op", {}, pending_confirmation=True)

    # Denial checks are untouched by the flag. (That a denial *is* recorded is
    # covered separately by test_denied_probe_still_consumes_quota_*.)
    with pytest.raises(PermissionDeniedError, match="not in allowed_domains"):
        policy.check_permission(
            "risky_op", {"url": "https://evil.test/x"}, pending_confirmation=True
        )

    # The replay that actually executes is recorded; with the denial above
    # that exhausts the budget of 2.
    policy.check_permission("risky_op", {}, confirmation_approved=True)
    with pytest.raises(PermissionDeniedError, match="Rate limit exceeded"):
        policy.check_permission("risky_op", {})


def test_denied_probe_still_consumes_quota_despite_pending_confirmation():
    """``pending_confirmation`` defers accounting because the call will stop
    at the executor's gate — but a *denial* is terminal regardless, and the
    executor returns before its own rate limiter is acquired, so skipping it
    here left rejected calls unbounded (PR #381 review)."""
    import pytest

    from agent_gantry.core.security import PermissionDeniedError, SecurityPolicy

    policy = SecurityPolicy(
        require_confirmation=[], allowed_domains=["example.com"], max_requests_per_minute=1
    )

    with pytest.raises(PermissionDeniedError, match="not in allowed_domains"):
        policy.check_permission(
            "risky", {"url": "https://evil.test/x"}, pending_confirmation=True
        )
    # The denial consumed the budget, so the next call is rate-limited rather
    # than being able to repeat forever.
    with pytest.raises(PermissionDeniedError, match="Rate limit exceeded"):
        policy.check_permission(
            "risky", {"url": "https://evil.test/x"}, pending_confirmation=True
        )


def test_clean_probes_remain_exempt_from_accounting():
    """The denial carve-out must not undo the probe exemption itself."""
    import pytest

    from agent_gantry.core.security import PermissionDeniedError, SecurityPolicy

    policy = SecurityPolicy(
        require_confirmation=[], allowed_domains=["example.com"], max_requests_per_minute=1
    )

    for _ in range(3):
        policy.check_permission(
            "risky", {"url": "https://example.com/ok"}, pending_confirmation=True
        )

    # …and the call that actually executes is still counted.
    policy.check_permission("risky", {"url": "https://example.com/ok"})
    with pytest.raises(PermissionDeniedError, match="Rate limit exceeded"):
        policy.check_permission("risky", {"url": "https://example.com/ok"})


def test_arguments_valid_false_charges_a_pattern_gated_call():
    """The policy defers its charge past its own confirmation gate so an
    approved replay isn't denied for the rest of the window. That is only
    right when a replay can happen: a call whose arguments already failed
    validation is terminal, so it must be charged where it stands
    (PR #381 review)."""
    import pytest

    from agent_gantry.core.security import (
        ConfirmationRequiredError,
        PermissionDeniedError,
        SecurityPolicy,
    )

    policy = SecurityPolicy(require_confirmation=["delete_*"], max_requests_per_minute=2)

    for _ in range(2):
        with pytest.raises(ConfirmationRequiredError):
            policy.check_permission("delete_thing", {}, arguments_valid=False)

    # The window is now full, so the third is denied rather than deferred.
    with pytest.raises(PermissionDeniedError, match="Rate limit"):
        policy.check_permission("delete_thing", {}, arguments_valid=False)


def test_arguments_valid_defaults_to_true_and_keeps_the_exemption():
    """Omitting the keyword must behave exactly as before it existed, so a
    valid probe stays free and its approved replay still fits the window."""
    import pytest

    from agent_gantry.core.security import (
        ConfirmationRequiredError,
        PermissionDeniedError,
        SecurityPolicy,
    )

    policy = SecurityPolicy(require_confirmation=["delete_*"], max_requests_per_minute=1)

    for _ in range(3):
        with pytest.raises(ConfirmationRequiredError):
            policy.check_permission("delete_thing", {})

    # None of those probes consumed the single slot, so the approval runs.
    policy.check_permission("delete_thing", {}, confirmation_approved=True)
    with pytest.raises(PermissionDeniedError, match="Rate limit"):
        policy.check_permission("delete_thing", {}, confirmation_approved=True)


def test_pattern_gated_execution_is_rate_limited_like_any_other():
    """Deferring a *probe*'s charge must not exempt the tool itself. Only the
    prompt is free: every call that actually clears the gate is counted, so a
    ``require_confirmation`` pattern cannot be used to make a sensitive tool
    unlimited (PR #381 review)."""
    import pytest

    from agent_gantry.core.security import (
        ConfirmationRequiredError,
        PermissionDeniedError,
        SecurityPolicy,
    )

    policy = SecurityPolicy(require_confirmation=["delete_*"], max_requests_per_minute=3)

    # Any number of unapproved probes: none executes, so none is charged.
    for _ in range(10):
        with pytest.raises(ConfirmationRequiredError):
            policy.check_permission("delete_thing", {})
    assert not policy._request_timestamps

    # Approved calls do execute, and are charged exactly like an ungated tool.
    for _ in range(3):
        policy.check_permission("delete_thing", {}, confirmation_approved=True)
    for _ in range(2):
        with pytest.raises(PermissionDeniedError, match="Rate limit"):
            policy.check_permission("delete_thing", {}, confirmation_approved=True)
    assert len(policy._request_timestamps) == 3
