/**
 * Agent-Gantry Documentation - Client-Side Search
 * Simple but effective search using a pre-built search index
 */

(function() {
  'use strict';

  // Search index will be populated by Jekyll or built at runtime
  let searchIndex = [];
  let searchInput = null;
  let searchResults = null;

  // ==================== Search Index Building ====================

  /**
   * Build search index from all documentation pages
   */
  async function buildSearchIndex() {
    // In a Jekyll site, we'd use a JSON file generated at build time
    // For now, we'll scrape the current page and navigation links

    const pages = [];

    // Add current page
    const currentContent = document.querySelector('.content-wrapper');
    if (currentContent) {
      pages.push({
        title: document.title,
        url: window.location.pathname,
        content: currentContent.textContent.toLowerCase()
      });
    }

    // Add linked pages from navigation
    const navLinks = document.querySelectorAll('.nav-link');
    navLinks.forEach(link => {
      const url = new URL(link.href).pathname;
      const title = link.textContent.trim();

      pages.push({
        title: title,
        url: url,
        content: title.toLowerCase() // Simplified - would fetch full content in production
      });
    });

    searchIndex = pages;
    console.log(`Search index built with ${searchIndex.length} pages`);
  }

  // ==================== Search Functionality ====================

  /**
   * Perform search and return ranked results
   */
  function search(query) {
    if (!query || query.length < 2) {
      return [];
    }

    const lowerQuery = query.toLowerCase();
    const queryTerms = lowerQuery.split(/\s+/);

    const results = searchIndex
      .map(page => {
        let score = 0;

        // Title match (highest weight)
        if (page.title.toLowerCase().includes(lowerQuery)) {
          score += 100;
        }

        // Individual term matches in title
        queryTerms.forEach(term => {
          if (page.title.toLowerCase().includes(term)) {
            score += 50;
          }
        });

        // Content matches
        queryTerms.forEach(term => {
          if (page.content.includes(term)) {
            score += 10;
          }
        });

        return {
          ...page,
          score: score
        };
      })
      .filter(page => page.score > 0)
      .sort((a, b) => b.score - a.score)
      .slice(0, 8); // Limit to top 8 results

    return results;
  }

  /**
   * Display search results
   */
  function displayResults(results) {
    if (!searchResults) return;

    // Clear previous results
    searchResults.innerHTML = '';

    if (results.length === 0) {
      searchResults.innerHTML = '<div class="search-no-results">No results found</div>';
      searchResults.style.display = 'block';
      return;
    }

    results.forEach(result => {
      const resultItem = document.createElement('a');
      resultItem.className = 'search-result-item';
      resultItem.href = result.url;

      const resultTitle = document.createElement('div');
      resultTitle.className = 'search-result-title';
      resultTitle.textContent = result.title;

      const resultUrl = document.createElement('div');
      resultUrl.className = 'search-result-url';
      resultUrl.textContent = result.url;

      resultItem.appendChild(resultTitle);
      resultItem.appendChild(resultUrl);
      searchResults.appendChild(resultItem);
    });

    searchResults.style.display = 'block';
  }

  /**
   * Hide search results
   */
  function hideResults() {
    if (searchResults) {
      searchResults.style.display = 'none';
    }
  }

  // ==================== Initialize Search ====================

  function initSearch() {
    searchInput = document.querySelector('.search-input');
    if (!searchInput) {
      console.log('Search input not found, skipping search initialization');
      return;
    }

    // Create results container if it doesn't exist
    searchResults = document.querySelector('.search-results');
    if (!searchResults) {
      searchResults = document.createElement('div');
      searchResults.className = 'search-results';
      searchResults.style.cssText = `
        position: absolute;
        top: 100%;
        left: 0;
        right: 0;
        background: white;
        border: 1px solid var(--border-color);
        border-radius: 6px;
        margin-top: 4px;
        box-shadow: var(--shadow-lg);
        max-height: 400px;
        overflow-y: auto;
        display: none;
        z-index: 1000;
      `;

      const searchContainer = searchInput.closest('.search-container');
      if (searchContainer) {
        searchContainer.style.position = 'relative';
        searchContainer.appendChild(searchResults);
      }
    }

    // Add search result item styles
    const style = document.createElement('style');
    style.textContent = `
      .search-result-item {
        display: block;
        padding: 12px 16px;
        border-bottom: 1px solid var(--border-color);
        text-decoration: none;
        color: var(--text-primary);
        transition: background-color 150ms;
      }

      .search-result-item:last-child {
        border-bottom: none;
      }

      .search-result-item:hover {
        background-color: var(--bg-secondary);
        text-decoration: none;
      }

      .search-result-title {
        font-weight: 600;
        margin-bottom: 4px;
        color: var(--text-primary);
      }

      .search-result-url {
        font-size: 0.75rem;
        color: var(--text-muted);
      }

      .search-no-results {
        padding: 16px;
        text-align: center;
        color: var(--text-muted);
      }
    `;
    document.head.appendChild(style);

    // Handle search input
    let searchTimeout;
    searchInput.addEventListener('input', function(e) {
      const query = e.target.value.trim();

      // Debounce search
      clearTimeout(searchTimeout);
      searchTimeout = setTimeout(() => {
        if (query.length >= 2) {
          const results = search(query);
          displayResults(results);
        } else {
          hideResults();
        }
      }, 200);
    });

    // Handle keyboard navigation in results
    searchInput.addEventListener('keydown', function(e) {
      if (!searchResults || searchResults.style.display === 'none') return;

      const items = searchResults.querySelectorAll('.search-result-item');
      if (items.length === 0) return;

      const activeItem = searchResults.querySelector('.search-result-item.active');
      let currentIndex = activeItem ? Array.from(items).indexOf(activeItem) : -1;

      if (e.key === 'ArrowDown') {
        e.preventDefault();
        currentIndex = (currentIndex + 1) % items.length;
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        currentIndex = currentIndex <= 0 ? items.length - 1 : currentIndex - 1;
      } else if (e.key === 'Enter') {
        e.preventDefault();
        if (activeItem) {
          activeItem.click();
        } else if (items[0]) {
          items[0].click();
        }
        return;
      } else if (e.key === 'Escape') {
        hideResults();
        searchInput.blur();
        return;
      } else {
        return;
      }

      // Update active item
      items.forEach(item => item.classList.remove('active'));
      items[currentIndex].classList.add('active');
      items[currentIndex].scrollIntoView({ block: 'nearest' });
    });

    // Close search results when clicking outside
    document.addEventListener('click', function(e) {
      if (searchInput && searchResults &&
          !searchInput.contains(e.target) &&
          !searchResults.contains(e.target)) {
        hideResults();
      }
    });

    // Build search index
    buildSearchIndex();

    console.log('Search initialized');
  }

  // ==================== Initialize ====================

  function init() {
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', init);
      return;
    }

    initSearch();
  }

  init();
})();
