// Delete chatbot confirmation
function deleteChatbot(chatbotId) {
    if (confirm('Are you sure you want to delete this chatbot? This action cannot be undone.')) {
        window.location.href = `/chatbot/delete/${chatbotId}`;
    }
}

document.addEventListener('DOMContentLoaded', function() {
    // --- Auto-hide alerts ---
    const alerts = document.querySelectorAll('.alert');
    alerts.forEach(alert => {
        setTimeout(() => {
            alert.style.transition = 'opacity 0.5s, transform 0.5s';
            alert.style.opacity = '0';
            alert.style.transform = 'translateY(-10px)';
            setTimeout(() => alert.remove(), 500);
        }, 5000);
    });

    // --- Search and Clear Button ---
    const searchInput = document.getElementById('searchInput');
    const clearButton = document.getElementById('clearSearch');

    // Combined search input handler
    searchInput.addEventListener('input', function() {
        // Show/hide clear button
        if (this.value.length > 0) {
            clearButton.style.display = 'flex';
        } else {
            clearButton.style.display = 'none';
        }
        // Filter chatbots
        filterChatbots();
    });

    // Clear search when button is clicked
    clearButton.addEventListener('click', function() {
        searchInput.value = '';
        clearButton.style.display = 'none';
        searchInput.focus();
        // Trigger input event to reset search results
        searchInput.dispatchEvent(new Event('input'));
    });

    // --- Filter ---
    const filterBtns = document.querySelectorAll('.filter-btn');
    filterBtns.forEach(btn => {
        btn.addEventListener('click', function() {
            filterBtns.forEach(b => b.classList.remove('active'));
            this.classList.add('active');
            filterChatbots();
        });
    });

    // --- Sort ---
    const sortSelect = document.getElementById('sortSelect');
    sortSelect.addEventListener('change', sortChatbots);


    // --- Sidebar menus (IMPROVED VERSION) ---
const botsMenu = document.getElementById('botsMenu');
const botsSubmenu = document.getElementById('botsSubmenu');
const accountMenu = document.getElementById('accountMenu');
const accountSubmenu = document.getElementById('accountSubmenu');

if (botsMenu && botsSubmenu) {
    botsMenu.addEventListener('click', function(e) {
        e.preventDefault();

        // Close account submenu if open
        if (accountSubmenu && accountSubmenu.classList.contains('open')) {
            accountSubmenu.classList.remove('open');
            accountMenu.classList.remove('expanded');
        }

        // Toggle bots submenu
        this.classList.toggle('expanded');
        botsSubmenu.classList.toggle('open');
    });
}

if (accountMenu && accountSubmenu) {
    accountMenu.addEventListener('click', function(e) {
        e.preventDefault();

        // Close bots submenu if open
        if (botsSubmenu && botsSubmenu.classList.contains('open')) {
            botsSubmenu.classList.remove('open');
            botsMenu.classList.remove('expanded');
        }

        // Toggle account submenu
        this.classList.toggle('expanded');
        accountSubmenu.classList.toggle('open');
    });
}

// Close submenus when clicking outside (ADD THIS NEW CODE)
document.addEventListener('click', function(e) {
    // Close bots submenu
    if (botsMenu && botsSubmenu && !e.target.closest('#botsMenu') && !e.target.closest('#botsSubmenu')) {
        botsSubmenu.classList.remove('open');
        botsMenu.classList.remove('expanded');
    }

    // Close account submenu
    if (accountMenu && accountSubmenu && !e.target.closest('#accountMenu') && !e.target.closest('#accountSubmenu')) {
        accountSubmenu.classList.remove('open');
        accountMenu.classList.remove('expanded');
    }
});
    // --- Mobile sidebar toggle ---
    const mobileMenuToggle = document.getElementById('mobileMenuToggle');
    const sidebar = document.getElementById('sidebar');

    mobileMenuToggle.addEventListener('click', function() {
        sidebar.classList.toggle('open');
    });

    document.addEventListener('click', function(e) {
        if (window.innerWidth <= 968) {
            if (!sidebar.contains(e.target) && !mobileMenuToggle.contains(e.target)) {
                sidebar.classList.remove('open');
            }
        }
    });

    // --- Animate cards on load ---
    const cards = document.querySelectorAll('.chatbot-card');
    cards.forEach((card, index) => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(20px)';
        setTimeout(() => {
            card.style.transition = 'opacity 0.5s, transform 0.5s';
            card.style.opacity = '1';
            card.style.transform = 'translateY(0)';
        }, index * 100);
    });
});


// --- Sort Function ---
function sortChatbots() {
    const sortValue = document.getElementById('sortSelect').value;
    const grid = document.querySelector('.chatbots-grid');
    const cards = Array.from(grid.querySelectorAll('.chatbot-card'));

    cards.sort((a, b) => {
        if (sortValue === 'name') {
            return a.querySelector('h3').textContent.localeCompare(b.querySelector('h3').textContent);
        }
        if (sortValue === 'name-desc') {
            return b.querySelector('h3').textContent.localeCompare(a.querySelector('h3').textContent);
        }

        if (sortValue === 'newest' || sortValue === 'oldest') {
            const dateA = new Date(a.dataset.createdAt);
            const dateB = new Date(b.dataset.createdAt);
            if (isNaN(dateA) || isNaN(dateB)) return 0;

            return sortValue === 'newest' ? dateB - dateA : dateA - dateB;
        }

        return 0;
    });

    cards.forEach(card => grid.appendChild(card));
}


// --- Filter Function ---
function filterChatbots() {
    const searchTerm = document.getElementById('searchInput').value.toLowerCase();
    const activeFilter = document.querySelector('.filter-btn.active').dataset.filter;
    const cards = document.querySelectorAll('.chatbot-card');
    const noResults = document.getElementById('noResults');
    let visibleCount = 0;

    cards.forEach(card => {
        const name = card.querySelector('h3').textContent.toLowerCase();
        const desc = card.querySelector('.chatbot-description')?.textContent.toLowerCase() || '';
        const status = card.querySelector('.status-badge').classList.contains('active') ? 'active' : 'inactive';

        const matchesSearch = name.includes(searchTerm) || desc.includes(searchTerm);
        const matchesFilter = (activeFilter === 'all') || (status === activeFilter);

        if (matchesSearch && matchesFilter) {
            card.style.display = '';
            visibleCount++;
        } else {
            card.style.display = 'none';
        }
    });

    noResults.style.display = visibleCount === 0 ? 'block' : 'none';
}