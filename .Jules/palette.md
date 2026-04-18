## 2026-04-17 - [External Link Decorative Icon Accessibility]
**Learning:** For accessibility in HTML/JS components, add `aria-hidden="true"` to decorative external link icons. Instead of appending a visually hidden `.sr-only` span, which can cause visual regressions if the CSS is missing or misapplied, prefer setting an `aria-label` directly on the link element (e.g., `link.setAttribute('aria-label', originalText + ' (opens in a new tab)');`).
**Action:** Update external link icon scripts to append the `aria-label` to the anchor tag itself, making sure to capture the original textContent before any new child nodes (like icons) are appended.

## 2026-04-18 - [Keyboard Navigation Focus State in Search Results]
**Learning:** Keyboard navigation (Up/Down arrows) added a dynamic `.active` class to search results via JavaScript, but there was no corresponding CSS styling. Consequently, screen reader or keyboard-only users couldn't see which item was focused. By appending `.search-result-item.active` to share the `.search-result-item:hover` styles, this was cleanly addressed without structural changes.
**Action:** When inspecting custom Javascript-driven keyboard navigation elements, always verify that the Javascript-toggled focus/active class is mirrored in the CSS to visually match the `:hover` state.

## 2026-04-21 - [Screen Reader Swallowing ARIA Labels]
**Learning:** Screen readers completely ignore an element's inner text if an `aria-label` is present. Therefore, if a link has existing text and you want to add contextual information for screen readers (like "(opens in a new tab)"), you must include the original text inside the `aria-label` instead of just appending context via a `.sr-only` span or using `aria-label` on a child element if that behavior could lead to confusing nesting. For `aria-label` appended dynamically, ensure you preserve the original `textContent`.
**Action:** Consistently set or append contextual string like "(opens in a new tab)" directly to the main element's `aria-label` attribute (incorporating its original text) rather than appending visually hidden child text nodes.
