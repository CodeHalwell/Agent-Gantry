## 2026-04-17 - [External Link Decorative Icon Accessibility]
**Learning:** For accessibility in HTML/JS components, add `aria-hidden="true"` to decorative external link icons. Instead of appending a visually hidden `.sr-only` span, which can cause visual regressions if the CSS is missing or misapplied, prefer setting an `aria-label` directly on the link element (e.g., `link.setAttribute('aria-label', originalText + ' (opens in a new tab)');`).
**Action:** Update external link icon scripts to append the `aria-label` to the anchor tag itself, making sure to capture the original textContent before any new child nodes (like icons) are appended.

## 2026-04-19 - [ARIA Labels with Visual Context]
**Learning:** For accessibility in HTML/JS components, screen readers will read the `aria-label` attribute on an element and completely ignore its inner text. Therefore, if you add a visual indicator via a hidden element (like appending `<span class="sr-only">(opens in a new tab)</span>`), it will not be announced if the parent element already has an `aria-label`.
**Action:** When adding context like "opens in a new tab" for external links, append that string directly to the existing `aria-label` attribute rather than adding visually hidden DOM elements to the link's text content.
