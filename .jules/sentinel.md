## 2025-04-20 - Fix Newline Bypass Vulnerability in Regex Validation

**Vulnerability:** Regular expressions used for validating SQL identifiers (table names) and tool names used `$` to match the end of the string. In Python's `re` module, `$` matches either the end of the string OR just before a trailing newline. This allowed a bypass where an identifier like `my_table\n` would pass validation but could potentially break logic or allow injection if concatenated directly.
**Learning:** Python's `re.match` and `re.search` default behavior for `$` is often misunderstood as strictly end-of-string.
**Prevention:** Always use `\Z` instead of `$` when you require a strict end-of-string match in Python regular expressions to prevent newline bypass attacks.
