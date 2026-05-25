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

## 2025-02-28 - Regex Bypass via Multiline Input
**Vulnerability:** The `validate_description` function used `re.search(pattern, desc, re.IGNORECASE)` to detect malicious patterns like `{{.*}}`. Because the `.*` wildcard does not match newline characters by default, attackers could bypass this validation by inserting newlines into their payloads (e.g., `{{\n malicious code \n}}`).
**Learning:** When using the `.*` wildcard to match content between delimiters for security validation, the dot (`.`) does not inherently match newline characters (`\n`). This allows multiline payloads to evade detection if the regular expression is not configured to treat the input as a single string encompassing newlines.
**Prevention:** Always use the `re.DOTALL` flag (often combined with others like `re.IGNORECASE`) in `re.search` when you need the wildcard `.` to match any character, including newlines, ensuring that multiline payloads are correctly identified and blocked.

## 2024-05-13 - [HIGH] Fix SQL injection bypass via unquoted identifiers
**Vulnerability:** The `PGVectorStore` implementation used string formatting to insert the `{self._table_name}` into raw SQL queries without double quotes (e.g., `CREATE TABLE IF NOT EXISTS {self._table_name}`). While the table name was validated against a strict regex, if validation were to fail or be bypassed, or if a table name matched a reserved SQL keyword, it could lead to SQL injection or syntax errors because PostgreSQL identifiers should be double-quoted when constructed dynamically.
**Learning:** Even when SQL identifiers (like table or column names) are validated using strict regular expressions, they must always be properly double-quoted when injected directly into raw SQL strings. This provides a critical second layer of defense (defense-in-depth) against injection vulnerabilities and prevents syntax errors from reserved keyword collisions.
**Prevention:** Always wrap dynamically injected SQL identifiers in double quotes (e.g., `"{table_name}"`) when using string interpolation or f-strings in Python for database queries.
## 2026-05-18 - [HIGH] Fix regex bypass via newline injection in tool identifier validation
**Vulnerability:** The `ToolDefinition.name` and `version` fields were validated using Pydantic's `Field(pattern=...)` which uses the Rust regex crate. The Rust regex engine's `$` anchor matches the end of the string OR the end of the line, meaning that if a trailing newline was present (e.g., `"valid_name
"`), the pattern validation would pass, potentially allowing log injection or HTTP header injection vulnerabilities. The `namespace` field lacked newline validation entirely.
**Learning:** In Pydantic v2, `Field(pattern=...)` uses the Rust `regex` crate, which does not support the `\Z` anchor (absolute end of string) and will raise a `SchemaError`. Furthermore, just checking `.endswith('
')` is insufficient security theater since newlines can be injected into the middle of the string or followed by spaces.
**Prevention:** To explicitly reject newlines and avoid security bypasses in Pydantic models, use a custom `@field_validator` checking `"
" in v or "" in v` across the entire string rather than relying on regex `$` or checking `.endswith('
')`.

## 2024-05-18 - [MEDIUM] Fix unhandled PermissionDeniedError in executor
**Vulnerability:** The `ExecutionEngine._check_security_policy` method caught `ConfirmationRequiredError` raised by `SecurityPolicy.check_permission` but failed to catch `PermissionDeniedError`. When a domain was restricted by the security policy or a rate limit was reached, an unhandled exception would bubble up instead of cleanly returning a failed `ToolResult`.
**Learning:** Security policy exceptions (like authorization and permission errors) must be exhaustively caught within the tool execution pipeline to ensure the engine gracefully and securely fails, logging the outcome without crashing the parent application or exposing raw stack traces.
**Prevention:** Always verify that every custom exception raised by a security or validation component is explicitly handled in the calling method, specifically when translating application exceptions into structured API or execution responses like `ToolResult`.

## 2024-05-24 - Fix SSRF / path traversal bypass in URL domain extraction
**Vulnerability:** The `SecurityPolicy` domain extraction relied on `urllib.parse.urlparse` to validate allowed domains, which allows bypasses via local file inclusion (like `file:///etc/passwd`) or malformed URLs (like `http:///etc/passwd`) when the hostname evaluates to `None` or an empty string, effectively ignoring the domain restriction entirely.
**Learning:** Checking `parsed.hostname` alone is insufficient when dealing with a list of domains because URLs without valid hostnames bypass the check by not adding any elements to the domains set, thereby making `_is_domain_allowed` not trigger an exception if there are no extracted domains.
**Prevention:** Explicitly detect and safely reject dangerous schemes (like `file`) and empty hostnames (where `parsed.netloc` is falsey or hostname evaluates to `None`) by returning a non-allowable domain marker (like `<invalid_domain>`) to ensure the security policy correctly fails the request.

## 2024-05-25 - Fix SSRF bypass via malformed URLs in domain validation
**Vulnerability:** An attacker could bypass the `allowed_domains` restriction in `SecurityPolicy` by providing malformed URLs like `http://@/etc/passwd` or `http://example.com@`.
**Learning:** `urllib.parse.urlparse` can return a non-empty `netloc` (e.g., `'@'` or `'example.com@'`) even when `hostname` is `None` or invalid. Checking `not parsed.netloc` instead of `not parsed.hostname` allowed these malformed URLs to skip the `<invalid_domain>` assignment, leaving the extracted domains set empty and bypassing the restriction check entirely.
**Prevention:** When enforcing SSRF and domain restrictions, explicitly check `parsed.hostname` (or handle it safely) instead of relying on `parsed.netloc`, as `netloc` may contain only userinfo or port components without a valid hostname.
