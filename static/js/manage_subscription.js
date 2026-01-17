function confirmCancel() {
            if (confirm('Are you sure you want to cancel your subscription? You will lose access to premium features at the end of your billing period.')) {
                window.location.href = "{{ url_for('cancel_subscription') }}";
            }
        }

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