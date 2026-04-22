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
