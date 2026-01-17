// Password visibility toggle
document.addEventListener('DOMContentLoaded', function() {
    const toggleButtons = document.querySelectorAll('.toggle-password');

    toggleButtons.forEach(button => {
        button.addEventListener('click', function() {
            const targetId = this.getAttribute('data-target');
            const input = document.getElementById(targetId);
            const eyeOpen = this.querySelector('.eye-open');
            const eyeClosed = this.querySelector('.eye-closed');

            if (input.type === 'password') {
                input.type = 'text';
                eyeOpen.style.display = 'none';
                eyeClosed.style.display = 'block';
            } else {
                input.type = 'password';
                eyeOpen.style.display = 'block';
                eyeClosed.style.display = 'none';
            }
        });
    });

    // Password confirmation validation
    const passwordForm = document.getElementById('passwordForm');
    if (passwordForm) {
        passwordForm.addEventListener('submit', function(e) {
            const newPassword = document.getElementById('new_password').value;
            const confirmPassword = document.getElementById('confirm_password').value;

            if (newPassword !== confirmPassword) {
                e.preventDefault();
                alert('Passwords do not match!');
                return false;
            }
        });
    }

    // Reset password form validation
    const resetPasswordForm = document.getElementById('resetPasswordForm');
    if (resetPasswordForm) {
        resetPasswordForm.addEventListener('submit', function(e) {
            const password = document.getElementById('password').value;
            const confirmPassword = document.getElementById('confirm_password').value;

            if (password !== confirmPassword) {
                e.preventDefault();
                alert('Passwords do not match!');
                return false;
            }
        });
    }
});

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