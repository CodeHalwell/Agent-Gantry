## 2024-05-24 - External links accessibility
**Learning:** Automatically appended visual indicators (like `↗`) for external links are read aloud by screen readers confusingly (e.g., "link text arrow up right").
**Action:** Always add `aria-hidden="true"` to the decorative external link icon and append visually hidden text (using `.sr-only` class) to explicitly announce "(opens in a new tab)" to screen reader users.
