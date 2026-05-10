## 2025-05-18 - Optimized String Matching in Hot Loops

**Learning:** Generator expressions inside `any()` for substring or character matching (e.g., `any(kw in text for kw in keywords)` or `any(ord(c) < 32 for c in text)`) incur significant overhead in hot loops. Converting these to pre-compiled regular expressions using `re.compile()` and `.search()` pushes the iteration logic down to C, providing substantial performance improvements.

**Action:** Whenever iterating over substrings or single characters for boolean checks in heavily executed code paths, evaluate if pre-compiled regular expressions can replace generator expressions. Additionally, always clean up temporary files created for benchmarking before committing.
