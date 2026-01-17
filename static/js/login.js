document.addEventListener('DOMContentLoaded', function() {
            const loginForm = document.getElementById('loginForm');
            const emailInput = document.getElementById('email');
            const passwordInput = document.getElementById('password');
            const togglePassword = document.getElementById('togglePassword');

            // Password toggle functionality
            togglePassword.addEventListener('click', function() {
                const type = passwordInput.getAttribute('type') === 'password' ? 'text' : 'password';
                passwordInput.setAttribute('type', type);

                // Toggle icon
                if (type === 'password') {
                    this.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path><circle cx="12" cy="12" r="3"></circle></svg>';
                } else {
                    this.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24"></path><line x1="1" y1="1" x2="23" y2="23"></line></svg>';
                }
            });

            // Form validation
            loginForm.addEventListener('submit', function(e) {
                let isValid = true;

                if (!validateEmail(emailInput.value)) {
                    showError(emailInput, 'Please enter a valid email address');
                    isValid = false;
                } else {
                    clearError(emailInput);
                }

                if (passwordInput.value.length < 6) {
                    showError(passwordInput, 'Password must be at least 6 characters');
                    isValid = false;
                } else {
                    clearError(passwordInput);
                }

                if (!isValid) {
                    e.preventDefault();
                }
            });

            emailInput.addEventListener('blur', function() {
                if (!validateEmail(this.value)) {
                    showError(this, 'Please enter a valid email address');
                } else {
                    clearError(this);
                }
            });

            passwordInput.addEventListener('input', function() {
                if (this.value.length > 0 && this.value.length < 6) {
                    showError(this, 'Password must be at least 6 characters');
                } else {
                    clearError(this);
                }
            });

            function validateEmail(email) {
                const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
                return re.test(email);
            }

            function showError(input, message) {
                clearError(input);
                input.style.borderColor = '#f56565';
                const errorDiv = document.createElement('div');
                errorDiv.className = 'error-message';
                errorDiv.textContent = message;
                input.parentElement.appendChild(errorDiv);
            }

            function clearError(input) {
                input.style.borderColor = '#e2e8f0';
                const errorDiv = input.parentElement.querySelector('.error-message');
                if (errorDiv) {
                    errorDiv.remove();
                }
            }

            const alerts = document.querySelectorAll('.alert');
            alerts.forEach(alert => {
                setTimeout(() => {
                    alert.style.transition = 'opacity 0.5s';
                    alert.style.opacity = '0';
                    setTimeout(() => alert.remove(), 500);
                }, 5000);
            });
        });