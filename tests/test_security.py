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
    from agent_gantry.core.security import SecurityPolicy, PermissionDeniedError
    import pytest

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
    from agent_gantry.core.security import SecurityPolicy, PermissionDeniedError
    import pytest

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
