/**
 * ============================================================================
 * PROFESSIONAL DEPLOY PAGE SCRIPT
 * ============================================================================
 * Enhanced functionality for deployment interface
 */

(function() {
    'use strict';

    // ========================================================================
    // CONFIGURATION
    // ========================================================================

    const CONFIG = window.CHATBOT_DATA || {
        id: null,
        name: 'Chatbot',
        embedCode: '',
        directLink: ''
    };

    console.log('%c🚀 Deploy Page Initialized', 'color: #4F46E5; font-size: 14px; font-weight: bold');
    console.log('Chatbot Data:', CONFIG);

    // ========================================================================
    // UTILITY FUNCTIONS
    // ========================================================================

    const Utils = {
        /**
         * Copy text to clipboard
         */
        async copyToClipboard(text) {
            try {
                await navigator.clipboard.writeText(text);
                return true;
            } catch (err) {
                // Fallback for older browsers
                return this.fallbackCopy(text);
            }
        },

        /**
         * Fallback copy method
         */
        fallbackCopy(text) {
            const textarea = document.createElement('textarea');
            textarea.value = text;
            textarea.style.position = 'fixed';
            textarea.style.opacity = '0';
            document.body.appendChild(textarea);
            textarea.select();

            try {
                document.execCommand('copy');
                document.body.removeChild(textarea);
                return true;
            } catch (err) {
                document.body.removeChild(textarea);
                console.error('Fallback copy failed:', err);
                return false;
            }
        },

        /**
         * Show button feedback
         */
        showButtonFeedback(button, success = true) {
            if (!button) return;

            const originalHTML = button.innerHTML;

            if (success) {
                button.classList.add('copied');

                setTimeout(() => {
                    button.classList.remove('copied');
                }, 2000);
            } else {
                button.style.background = '#EF4444';

                setTimeout(() => {
                    button.style.background = '';
                }, 2000);
            }
        },

        /**
         * Generate QR Code URL
         */
        generateQRCodeURL(text) {
            const size = 300;
            return `https://api.qrserver.com/v1/create-qr-code/?size=${size}x${size}&data=${encodeURIComponent(text)}`;
        },

        /**
         * Download file
         */
        downloadFile(url, filename) {
            const link = document.createElement('a');
            link.href = url;
            link.download = filename;
            link.target = '_blank';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }
    };

    // ========================================================================
    // COPY FUNCTIONS
    // ========================================================================

    /**
     * Copy embed code
     */
    window.copyEmbedCode = async function() {
        const codeElement = document.getElementById('embedCode');
        if (!codeElement) return;

        const code = codeElement.textContent;
        const button = event.target.closest('.btn-copy');

        console.log('📋 Copying embed code...');

        const success = await Utils.copyToClipboard(code);

        if (success) {
            console.log('✅ Embed code copied!');
            Utils.showButtonFeedback(button, true);

            // Track analytics (optional)
            if (typeof gtag !== 'undefined') {
                gtag('event', 'copy_embed_code', {
                    chatbot_id: CONFIG.id,
                    chatbot_name: CONFIG.name
                });
            }
        } else {
            console.error('❌ Copy failed');
            alert('Failed to copy. Please try manually selecting and copying the code.');
        }
    };

    /**
     * Copy direct link
     */
    window.copyDirectLink = async function() {
        const linkInput = document.getElementById('directLink');
        if (!linkInput) return;

        const link = linkInput.value;
        const button = event.target.closest('.btn-copy');

        console.log('🔗 Copying direct link...');

        const success = await Utils.copyToClipboard(link);

        if (success) {
            console.log('✅ Direct link copied!');
            Utils.showButtonFeedback(button, true);

            // Track analytics
            if (typeof gtag !== 'undefined') {
                gtag('event', 'copy_direct_link', {
                    chatbot_id: CONFIG.id,
                    chatbot_name: CONFIG.name
                });
            }
        } else {
            console.error('❌ Copy failed');
            alert('Failed to copy link. Please try manually.');
        }
    };

    // ========================================================================
    // SHARE FUNCTIONS
    // ========================================================================

    /**
     * Share on social media
     */
    window.shareOnSocial = function() {
        const url = CONFIG.directLink;
        const text = `Check out my AI chatbot: ${CONFIG.name}`;

        // Check if Web Share API is available
        if (navigator.share) {
            navigator.share({
                title: CONFIG.name,
                text: text,
                url: url
            }).then(() => {
                console.log('✅ Shared successfully');
            }).catch((err) => {
                console.log('Share cancelled or failed:', err);
            });
        } else {
            // Fallback: Show share options
            showShareModal(url, text);
        }
    };

    /**
     * Show share modal (fallback)
     */
    function showShareModal(url, text) {
        const encodedUrl = encodeURIComponent(url);
        const encodedText = encodeURIComponent(text);

        const shareOptions = [
            {
                name: 'Twitter',
                url: `https://twitter.com/intent/tweet?text=${encodedText}&url=${encodedUrl}`,
                color: '#1DA1F2'
            },
            {
                name: 'Facebook',
                url: `https://www.facebook.com/sharer/sharer.php?u=${encodedUrl}`,
                color: '#4267B2'
            },
            {
                name: 'LinkedIn',
                url: `https://www.linkedin.com/sharing/share-offsite/?url=${encodedUrl}`,
                color: '#0077B5'
            },
            {
                name: 'WhatsApp',
                url: `https://wa.me/?text=${encodedText}%20${encodedUrl}`,
                color: '#25D366'
            }
        ];

        // Create modal
        const modal = document.createElement('div');
        modal.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.5);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 10000;
        `;

        const modalContent = document.createElement('div');
        modalContent.style.cssText = `
            background: white;
            padding: 32px;
            border-radius: 16px;
            max-width: 400px;
            width: 90%;
        `;

        modalContent.innerHTML = `
            <h3 style="margin-bottom: 20px; font-size: 20px; color: #1F2937;">Share Chatbot</h3>
            <div style="display: flex; flex-direction: column; gap: 12px;">
                ${shareOptions.map(option => `
                    <a href="${option.url}"
                       target="_blank"
                       style="display: flex; align-items: center; gap: 12px; padding: 12px; background: ${option.color}; color: white; text-decoration: none; border-radius: 8px; font-weight: 600;">
                        ${option.name}
                    </a>
                `).join('')}
            </div>
            <button onclick="this.closest('.share-modal').remove()"
                    style="width: 100%; margin-top: 16px; padding: 12px; background: #F3F4F6; border: none; border-radius: 8px; font-weight: 600; cursor: pointer;">
                Close
            </button>
        `;

        modal.className = 'share-modal';
        modal.appendChild(modalContent);
        document.body.appendChild(modal);

        // Close on outside click
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                modal.remove();
            }
        });
    }

    // ========================================================================
    // QR CODE FUNCTIONS
    // ========================================================================

    /**
     * Download QR Code
     */
    window.downloadQRCode = function() {
        console.log('📱 Generating QR Code...');

        const qrUrl = Utils.generateQRCodeURL(CONFIG.directLink);
        const filename = `${CONFIG.name.replace(/\s+/g, '_')}_QRCode.png`;

        Utils.downloadFile(qrUrl, filename);

        console.log('✅ QR Code download initiated');

        // Track analytics
        if (typeof gtag !== 'undefined') {
            gtag('event', 'download_qr_code', {
                chatbot_id: CONFIG.id,
                chatbot_name: CONFIG.name
            });
        }
    };

    // ========================================================================
    // KEYBOARD SHORTCUTS
    // ========================================================================

    document.addEventListener('keydown', function(e) {
        // Ctrl/Cmd + Shift + C: Copy embed code
        if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === 'C') {
            e.preventDefault();
            const copyBtn = document.querySelector('.code-block .btn-copy');
            if (copyBtn) copyBtn.click();
            console.log('⌨️ Keyboard shortcut: Copy embed code');
        }

        // Ctrl/Cmd + Shift + L: Copy link
        if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === 'L') {
            e.preventDefault();
            const linkCopyBtn = document.querySelector('.link-input-group .btn-copy');
            if (linkCopyBtn) linkCopyBtn.click();
            console.log('⌨️ Keyboard shortcut: Copy link');
        }
    });

    // ========================================================================
    // SIDEBAR MENU (if present)
    // ========================================================================

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

// Close submenus when clicking outside
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

    // ========================================================================
    // ANALYTICS TRACKING
    // ========================================================================

    /**
     * Track page view
     */
    if (typeof gtag !== 'undefined') {
        gtag('event', 'page_view', {
            page_title: 'Deploy Chatbot',
            page_location: window.location.href,
            chatbot_id: CONFIG.id,
            chatbot_name: CONFIG.name
        });
    }

    // ========================================================================
    // INITIALIZATION
    // ========================================================================

    console.log('✅ Deploy page ready');
    console.log('Keyboard shortcuts:');
    console.log('  - Ctrl/Cmd + Shift + C: Copy embed code');
    console.log('  - Ctrl/Cmd + Shift + L: Copy direct link');

})();