/**
 * Agent-Gantry Documentation - Navigation & Interactivity
 */

(function() {
  'use strict';

  // ==================== Mobile Menu Toggle ====================

  function initMobileMenu() {
    const toggleButton = document.querySelector('.mobile-menu-toggle');
    const sidebar = document.querySelector('.sidebar');

    if (toggleButton && sidebar) {
      toggleButton.addEventListener('click', function() {
        sidebar.classList.toggle('open');

        // Update aria-expanded for accessibility
        const isExpanded = sidebar.classList.contains('open');
        toggleButton.setAttribute('aria-expanded', isExpanded);
      });

      // Close sidebar when clicking outside on mobile
      document.addEventListener('click', function(event) {
        const isMobile = window.innerWidth <= 768;
        if (isMobile &&
            !sidebar.contains(event.target) &&
            !toggleButton.contains(event.target) &&
            sidebar.classList.contains('open')) {
          sidebar.classList.remove('open');
          toggleButton.setAttribute('aria-expanded', 'false');
        }
      });
    }
  }

  // ==================== Active Navigation Highlighting ====================

  function highlightActiveNav() {
    const currentPath = window.location.pathname;
    const navLinks = document.querySelectorAll('.nav-link');

    navLinks.forEach(link => {
      const linkPath = new URL(link.href).pathname;

      // Exact match or starts with for sub-pages
      if (currentPath === linkPath || currentPath.startsWith(linkPath + '/')) {
        link.classList.add('active');

        // Expand parent section if in submenu
        const parentMenu = link.closest('.nav-submenu');
        if (parentMenu) {
          const parentSection = parentMenu.closest('.nav-section');
          if (parentSection) {
            parentSection.classList.add('expanded');
          }
        }
      } else {
        link.classList.remove('active');
      }
    });
  }

  // ==================== Smooth Scroll with Offset ====================

  function initSmoothScroll() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
      anchor.addEventListener('click', function(e) {
        const href = this.getAttribute('href');

        // Skip if href is just '#'
        if (href === '#') return;

        const target = document.querySelector(href);
        if (target) {
          e.preventDefault();

          const headerOffset = 80; // Account for fixed header
          const elementPosition = target.getBoundingClientRect().top;
          const offsetPosition = elementPosition + window.pageYOffset - headerOffset;

          window.scrollTo({
            top: offsetPosition,
            behavior: 'smooth'
          });

          // Update URL without triggering scroll
          history.pushState(null, '', href);
        }
      });
    });
  }

  // ==================== Table of Contents Generator ====================

  function generateTableOfContents() {
    const content = document.querySelector('.content-wrapper');
    const tocContainer = document.querySelector('.table-of-contents');

    if (!content || !tocContainer) return;

    const headings = content.querySelectorAll('h2, h3, h4');
    if (headings.length === 0) return;

    const tocList = document.createElement('ul');
    tocList.className = 'toc-list';

    headings.forEach((heading, index) => {
      // Add ID if not present
      if (!heading.id) {
        heading.id = 'heading-' + index;
      }

      const level = parseInt(heading.tagName.substring(1));
      const listItem = document.createElement('li');
      listItem.className = 'toc-item toc-level-' + level;

      const link = document.createElement('a');
      link.href = '#' + heading.id;
      link.textContent = heading.textContent;
      link.className = 'toc-link';

      listItem.appendChild(link);
      tocList.appendChild(listItem);
    });

    tocContainer.appendChild(tocList);
  }

  // ==================== Code Copy Buttons ====================

  function initCodeCopyButtons() {
    const codeBlocks = document.querySelectorAll('pre code');

    codeBlocks.forEach(block => {
      const pre = block.parentElement;

      // Wrap pre in a container if not already wrapped
      if (!pre.parentElement.classList.contains('code-block-wrapper')) {
        const wrapper = document.createElement('div');
        wrapper.className = 'code-block-wrapper';
        pre.parentNode.insertBefore(wrapper, pre);
        wrapper.appendChild(pre);
      }

      // Create copy button
      const button = document.createElement('button');
      button.className = 'copy-button';
      button.textContent = 'Copy';
      button.setAttribute('aria-label', 'Copy code to clipboard');

      button.addEventListener('click', async function() {
        try {
          await navigator.clipboard.writeText(block.textContent);
          button.textContent = 'Copied!';
          button.classList.add('copied');

          setTimeout(() => {
            button.textContent = 'Copy';
            button.classList.remove('copied');
          }, 2000);
        } catch (err) {
          console.error('Failed to copy code:', err);
          button.textContent = 'Failed';

          setTimeout(() => {
            button.textContent = 'Copy';
          }, 2000);
        }
      });

      pre.parentElement.appendChild(button);
    });
  }

  // ==================== Scroll Progress Indicator ====================

  function initScrollProgress() {
    const progressBar = document.createElement('div');
    progressBar.className = 'scroll-progress';
    progressBar.style.cssText = `
      position: fixed;
      top: 0;
      left: 0;
      width: 0%;
      height: 3px;
      background: linear-gradient(90deg, #2563eb, #10b981);
      z-index: 1000;
      transition: width 100ms ease-out;
    `;
    document.body.appendChild(progressBar);

    window.addEventListener('scroll', function() {
      const windowHeight = document.documentElement.scrollHeight - document.documentElement.clientHeight;
      const scrolled = (window.scrollY / windowHeight) * 100;
      progressBar.style.width = scrolled + '%';
    });
  }

  // ==================== Keyboard Shortcuts ====================

  function initKeyboardShortcuts() {
    document.addEventListener('keydown', function(e) {
      // Cmd/Ctrl + K to focus search
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        const searchInput = document.querySelector('.search-input');
        if (searchInput) {
          searchInput.focus();
        }
      }
    });
  }

  // ==================== External Link Icons ====================

  function markExternalLinks() {
    const links = document.querySelectorAll('a[href^="http"]');

    links.forEach(link => {
      // Check if link is external
      const isExternal = !link.href.includes(window.location.hostname);

      if (isExternal) {
        link.setAttribute('target', '_blank');
        link.setAttribute('rel', 'noopener noreferrer');

        // Add visual indicator
        if (!link.querySelector('.external-icon')) {
          const icon = document.createElement('span');
          icon.className = 'external-icon';
          icon.innerHTML = ' ↗';
          icon.style.fontSize = '0.8em';
          icon.style.opacity = '0.6';
          link.appendChild(icon);
        }
      }
    });
  }

  // ==================== Collapsible Sections ====================

  function initCollapsibleSections() {
    const collapsibles = document.querySelectorAll('.collapsible');

    collapsibles.forEach(section => {
      const header = section.querySelector('.collapsible-header');
      const content = section.querySelector('.collapsible-content');

      if (header && content) {
        header.addEventListener('click', function() {
          const isOpen = section.classList.contains('open');
          section.classList.toggle('open');

          // Animate height
          if (isOpen) {
            content.style.maxHeight = null;
          } else {
            content.style.maxHeight = content.scrollHeight + 'px';
          }
        });
      }
    });
  }

  // ==================== Heading Anchor Links ====================

  function addHeadingAnchors() {
    const headings = document.querySelectorAll('.content-wrapper h2, .content-wrapper h3, .content-wrapper h4');

    headings.forEach(heading => {
      if (!heading.id) {
        heading.id = heading.textContent
          .toLowerCase()
          .replace(/[^\w\s-]/g, '')
          .replace(/\s+/g, '-');
      }

      const anchor = document.createElement('a');
      anchor.className = 'heading-anchor';
      anchor.href = '#' + heading.id;
      anchor.innerHTML = '#';
      anchor.title = 'Link to this section';
      anchor.style.cssText = `
        margin-left: 8px;
        opacity: 0;
        transition: opacity 150ms;
        text-decoration: none;
        color: var(--text-muted);
      `;

      heading.appendChild(anchor);

      heading.addEventListener('mouseenter', function() {
        anchor.style.opacity = '1';
      });

      heading.addEventListener('mouseleave', function() {
        anchor.style.opacity = '0';
      });
    });
  }

  // ==================== Initialize All Features ====================

  function init() {
    // Wait for DOM to be fully loaded
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', init);
      return;
    }

    initMobileMenu();
    highlightActiveNav();
    initSmoothScroll();
    generateTableOfContents();
    initCodeCopyButtons();
    initScrollProgress();
    initKeyboardShortcuts();
    markExternalLinks();
    initCollapsibleSections();
    addHeadingAnchors();

    console.log('Agent-Gantry Documentation - Navigation initialized');
  }

  // Start initialization
  init();
})();
