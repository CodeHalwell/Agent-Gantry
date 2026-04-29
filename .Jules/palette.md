## 2026-04-17 - [External Link Decorative Icon Accessibility]
**Learning:** For accessibility in HTML/JS components, add `aria-hidden="true"` to decorative external link icons. Instead of appending a visually hidden `.sr-only` span, which can cause visual regressions if the CSS is missing or misapplied, prefer setting an `aria-label` directly on the link element (e.g., `link.setAttribute('aria-label', originalText + ' (opens in a new tab)');`).
**Action:** Update external link icon scripts to append the `aria-label` to the anchor tag itself, making sure to capture the original textContent before any new child nodes (like icons) are appended.

## 2026-04-18 - [Keyboard Navigation Focus State in Search Results]
**Learning:** Keyboard navigation (Up/Down arrows) added a dynamic `.active` class to search results via JavaScript, but there was no corresponding CSS styling. Consequently, screen reader or keyboard-only users couldn't see which item was focused. By appending `.search-result-item.active` to share the `.search-result-item:hover` styles, this was cleanly addressed without structural changes.
**Action:** When inspecting custom Javascript-driven keyboard navigation elements, always verify that the Javascript-toggled focus/active class is mirrored in the CSS to visually match the `:hover` state.

## 2026-04-21 - [Screen Reader Swallowing ARIA Labels]
**Learning:** Screen readers completely ignore an element's inner text if an `aria-label` is present. Therefore, if a link has existing text and you want to add contextual information for screen readers (like "(opens in a new tab)"), you must include the original text inside the `aria-label` instead of just appending context via a `.sr-only` span or using `aria-label` on a child element if that behavior could lead to confusing nesting. For `aria-label` appended dynamically, ensure you preserve the original `textContent`.
**Action:** Consistently set or append contextual string like "(opens in a new tab)" directly to the main element's `aria-label` attribute (incorporating its original text) rather than appending visually hidden child text nodes.

## 2026-04-23 - [Screen Reader Announcements for Dynamic Content]
**Learning:** For screen readers to announce dynamic content like search results when focus remains in an input field, you must use an `aria-live` region. Additionally, when using arrow keys to navigate a custom dropdown, you must manually update the `aria-live` region with the active item's text.
**Action:** Inject a visually hidden `aria-live="polite"` element and update its `textContent` when dynamic UI regions change state or list navigation occurs.

## 2024-04-24 - Accessibility for Custom Interactive Elements
**Learning:** Collapsible sections in `docs/assets/js/navigation.js` were built using standard `<div>` elements with only `click` event listeners. This entirely broke keyboard navigation and screen reader support, as non-semantic tags lack focusability (`tabindex="0"`) and default keyboard activation (`Enter`/`Space`).
**Action:** When building custom interactive components like collapsibles or dropdowns with non-interactive HTML elements (div/span), always explicitly add `role="button"`, `tabindex="0"`, `aria-expanded` state, and listen to `keydown` events for `Enter` and `Space` keys to restore native behavior.

## 2026-04-25 - [Keyboard Accessibility for JavaScript-Toggled Visibility]
**Learning:** Elements (like heading anchors) that are only made visible on `mouseenter` become invisible traps for keyboard navigators, who cannot see what element has currently received focus when tabbing. Additionally, symbols like "#" used as links are announced poorly (e.g., "number") by screen readers unless given contextual `aria-label`s.
**Action:** Always provide corresponding `focus` and `blur` event handlers on focusable elements if their visibility is dynamically toggled via `mouseenter`/`mouseleave`. Furthermore, ensure symbol-only links are given descriptive `aria-label`s capturing their functional context (e.g., "Link to section: [heading text]").

## 2026-04-29 - [Keyboard Focus Management with Programmatic Scrolling]
**Learning:** Intercepting anchor clicks (like "Skip to main content") with `e.preventDefault()` to apply smooth scrolling breaks native focus movement. If focus remains on the clicked link, subsequent `Tab` presses will not start from the target element, rendering the skip link ineffective for keyboard users.
**Action:** Always manually move focus to the target element (`target.focus({ preventScroll: true })`) and ensure it's focusable by temporarily setting `tabindex="-1"` when implementing programmatic smooth scrolling for in-page anchors.
