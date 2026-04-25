## 2024-05-10 - [Security Bypass]
**Vulnerability:** The SecurityPolicy.check_permission method iterates through the top-level values of the arguments dictionary, skipping them if they are not strings. This allows bypassing the allowed_domains check by passing a domain in a list or nested dictionary.
**Learning:** Argument validation must recursively check all values in the arguments structure (lists, dicts, etc.) instead of only checking the top-level string values. This is especially important for tools that might accept lists of URLs or objects with nested properties.
**Prevention:** Implement a recursive extraction method to traverse the entire argument structure when searching for domains, ensuring deep objects and arrays are fully evaluated against the policy.

## 2024-04-17 - Fix URL parsing bypass in domain extraction
**Vulnerability:** The `_extract_domains` function improperly parsed protocol-relative URLs using a regex `//([a-zA-Z0-9][-a-zA-Z0-9.]*\.[a-zA-Z]{2,})`. This allowed basic authentication payloads like `//github.com@evil.com` to extract the `github.com` username instead of the `evil.com` host, bypassing domain restrictions.
**Learning:** Always use standard URL parsing libraries (`urllib.parse`) instead of custom regex to extract hostnames from URLs.
**Prevention:** Use a regex to match the full URL string, then pass it to `urllib.parse.urlparse` to handle authentication components securely.
## 2024-05-11 - [CRITICAL] Fix SQL injection bypass via trailing newline in identifiers
**Vulnerability:** The `_validate_sql_identifier` and `validate_name` functions used `$` in their validation regular expressions (e.g., `re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", value)`). In Python's `re` module, `$` matches the end of the string or just before a trailing newline. This could allow an attacker to bypass validation and potentially perform SQL injection or logical bypasses by appending `\n` to an identifier (e.g., `valid_name\nDROP TABLE tools;`).
**Learning:** In Python regex, always use `\Z` instead of `$` to strictly match the absolute end of the string when validating identifiers or potentially unsafe inputs, as `$` allows trailing newlines which can lead to bypass vulnerabilities.
**Prevention:** Always use `\Z` anchor for absolute end-of-string matching in security-critical regular expressions in Python.

## 2025-02-24 - SSRF Bypass via URL Parsing Discrepancies
**Vulnerability:** The `SecurityPolicy` domain extraction relied on `urllib.parse.urlparse` to validate allowed domains. This was vulnerable to SSRF because `urlparse` doesn't normalize backslashes to forward slashes (unlike requests or web browsers). A URL like `http://evil.com\@example.com` would have `example.com` as the parsed hostname, bypassing domain restrictions, but would actually be requested against `evil.com`. Double URL encoding (`%255C`) and port injection (`http://example.com:evil.com`) could also bypass it.
**Learning:** Security controls that rely on URL parsing must use exactly the same parsing logic as the underlying client making the request. If the client normalizes `\` to `/`, the validation must do the same. If the client handles auth encoding differently, validation must match.
**Prevention:** Always normalize URLs by fully decoding them (repeatedly to catch double-encoding) and normalizing backslashes to forward slashes *before* validating the hostname. Additionally, handle `urlparse` exceptions explicitly, particularly `ValueError` on port extraction, to ensure malformed URLs default to denying access rather than returning partial/bypassed fields.

## 2024-05-13 - [CRITICAL] Fix regex bypass with newline characters in validate_description
**Vulnerability:** The `validate_description` function checked for suspicious patterns in tool descriptions using `re.search` without the `re.DOTALL` flag. This allowed bypasses using newline characters (e.g., `{{\n payload \n}}`), as the `.` wildcard does not match newlines by default. This could lead to template injection or XSS bypasses.
**Learning:** When using the regex wildcard `.*` to match content between delimiters for security validation (e.g., `{{.*}}`), the `re.DOTALL` flag must be used to prevent attackers from bypassing the check by injecting newline characters.
**Prevention:** Use `re.DOTALL` flag in `re.search` when performing security validation that involves matching across potential newline boundaries.
