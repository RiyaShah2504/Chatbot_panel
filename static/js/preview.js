/**
 * ============================================================================
 * ✅ FIXED: Bot Text Color Now Respects Configuration
 * ============================================================================
 */

const config = chatbotConfig;

console.log('🤖 Chatbot Preview Initialized');
console.log('Config:', config);

if (typeof config.welcomeButtons === 'string') {
    try {
        config.welcomeButtons = JSON.parse(config.welcomeButtons);
    } catch (e) {
        console.error('Failed to parse welcome buttons:', e);
        config.welcomeButtons = [];
    }
}

if (!Array.isArray(config.welcomeButtons)) {
    config.welcomeButtons = [];
}

console.log('✅ Welcome Buttons loaded:', config.welcomeButtons.length);
console.log('📋 Buttons data:', config.welcomeButtons);

// ===== HELPER FUNCTIONS =====

function getRobotIconSVG(themeColor) {
    return `
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" width="100%" height="100%">
            <rect x="16" y="24" width="32" height="28" rx="6" fill="${themeColor}"/>
            <line x1="32" y1="16" x2="32" y2="24" stroke="${themeColor}" stroke-width="2.5" stroke-linecap="round"/>
            <circle cx="32" cy="14" r="3" fill="${themeColor}"/>
            <circle cx="24" cy="34" r="3.5" fill="white"/>
            <circle cx="40" cy="34" r="3.5" fill="white"/>
            <path d="M 24 44 Q 32 48 40 44" stroke="white" stroke-width="2.5" fill="none" stroke-linecap="round"/>
            <rect x="10" y="30" width="6" height="14" rx="3" fill="${themeColor}" opacity="0.8"/>
            <rect x="48" y="30" width="6" height="14" rx="3" fill="${themeColor}" opacity="0.8"/>
            <circle cx="24" cy="46" r="1.5" fill="white" opacity="0.6"/>
            <circle cx="32" cy="46" r="1.5" fill="white" opacity="0.6"/>
            <circle cx="40" cy="46" r="1.5" fill="white" opacity="0.6"/>
        </svg>
    `;
}

function adjustBrightness(color, percent) {
    let r, g, b;

    if (color.startsWith('#')) {
        const hex = color.replace('#', '');
        r = parseInt(hex.substring(0, 2), 16);
        g = parseInt(hex.substring(2, 4), 16);
        b = parseInt(hex.substring(4, 6), 16);
    } else if (color.startsWith('rgb')) {
        const rgb = color.match(/\d+/g);
        r = parseInt(rgb[0]);
        g = parseInt(rgb[1]);
        b = parseInt(rgb[2]);
    } else {
        return color;
    }

    r = Math.max(0, Math.min(255, r + (r * percent / 100)));
    g = Math.max(0, Math.min(255, g + (g * percent / 100)));
    b = Math.max(0, Math.min(255, b + (b * percent / 100)));

    return `rgb(${Math.round(r)}, ${Math.round(g)}, ${Math.round(b)})`;
}

function createBotAvatar() {
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.style.display = 'flex';
    avatar.style.alignItems = 'center';
    avatar.style.justifyContent = 'center';

    if (config.botAvatar) {
        const img = document.createElement('img');
        img.src = config.botAvatar;
        img.alt = config.botName;
        img.style.width = '100%';
        img.style.height = '100%';
        img.style.objectFit = 'cover';
        img.style.borderRadius = '50%';
        avatar.appendChild(img);
        avatar.style.background = 'white';
        avatar.style.border = '2px solid #e2e8f0';
    } else {
        avatar.style.background = 'white';
        avatar.style.border = `2px solid ${config.themeColor}`;
        avatar.innerHTML = getRobotIconSVG(config.themeColor);
    }

    return avatar;
}

// ===== MAIN APPLICATION =====
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 DOM Content Loaded - Initializing preview...');

    const userInput = document.getElementById('userInput');
    const sendBtn = document.getElementById('sendBtn');
    const chatMessages = document.getElementById('chatMessages');
    const chatHeader = document.getElementById('chatHeader');
    const chatAvatar = document.getElementById('chatAvatar');

    let sessionId = null;
    let currentSubmenuParent = null;

    applyThemeColors();
    displayWelcomeMessage();

    if (sendBtn) {
        sendBtn.addEventListener('click', function(e) {
            e.preventDefault();
            sendMessage();
        });
    }

    if (userInput) {
        userInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
    }

    function applyThemeColors() {
        console.log('🎨 Applying theme colors');
        console.log('   Theme Color:', config.themeColor);
        console.log('   Bot Message BG:', config.botMessageColor);
        console.log('   Bot Text Color:', config.botTextColor);
        console.log('   User Text Color:', config.userTextColor);

        if (chatHeader) {
            chatHeader.style.background = `linear-gradient(135deg, ${config.themeColor} 0%, ${adjustBrightness(config.themeColor, -20)} 100%)`;
        }

        if (chatMessages) {
            chatMessages.style.background = config.chatBackgroundColor;
        }

        if (sendBtn) {
            sendBtn.style.background = `linear-gradient(135deg, ${config.themeColor} 0%, ${adjustBrightness(config.themeColor, -15)} 100%)`;
        }

        if (chatAvatar && !config.botAvatar) {
            chatAvatar.style.background = 'white';
            chatAvatar.style.border = `2px solid ${config.themeColor}`;
            chatAvatar.innerHTML = getRobotIconSVG(config.themeColor);
        } else if (chatAvatar && config.botAvatar) {
            chatAvatar.style.background = 'white';
            chatAvatar.style.border = '2px solid #e2e8f0';
            chatAvatar.innerHTML = `<img src="${config.botAvatar}" alt="${config.botName}" style="width: 100%; height: 100%; object-fit: cover; border-radius: 50%;">`;
        }

        console.log('✅ Theme colors applied successfully');
    }

    function displayWelcomeMessage() {
        if (!chatMessages) return;

        console.log('💬 Displaying welcome message...');

        const messageDiv = document.createElement('div');
        messageDiv.className = 'message bot-message';

        const avatar = createBotAvatar();
        messageDiv.appendChild(avatar);

        const wrapper = document.createElement('div');
        wrapper.className = 'message-wrapper';

        const content = document.createElement('div');
        content.className = 'message-content';
        content.textContent = config.welcomeMessage;
        // ✅ FIX: Use config colors
        content.style.background = config.botMessageColor;
        content.style.color = config.botTextColor;
        wrapper.appendChild(content);

        if (config.welcomeButtons && config.welcomeButtons.length > 0) {
            const buttonsContainer = document.createElement('div');
            buttonsContainer.className = 'welcome-buttons';

            config.welcomeButtons.forEach((button, index) => {
                console.log(`🔘 Creating button ${index + 1}:`, button.text);

                const btn = document.createElement('button');
                btn.className = 'welcome-button';
                btn.textContent = button.text || 'Button';
                btn.style.borderColor = config.themeColor;
                btn.style.color = config.themeColor;
                btn.style.background = 'white';

                if (button.has_submenu && button.submenu_items && button.submenu_items.length > 0) {
                    btn.classList.add('has-submenu');
                    const arrow = document.createElement('span');
                    arrow.className = 'submenu-arrow';
                    arrow.textContent = ' ▼';
                    btn.appendChild(arrow);
                    console.log(`   └─ Has ${button.submenu_items.length} submenu items`);
                }

                btn.addEventListener('mouseenter', function() {
                    this.style.background = config.themeColor;
                    this.style.color = 'white';
                });
                btn.addEventListener('mouseleave', function() {
                    this.style.background = 'white';
                    this.style.color = config.themeColor;
                });

                btn.addEventListener('click', function() {
                    handleButtonClick(button);
                });

                buttonsContainer.appendChild(btn);
            });

            wrapper.appendChild(buttonsContainer);
            console.log(`✅ Created ${config.welcomeButtons.length} main buttons`);
        }

        messageDiv.appendChild(wrapper);
        chatMessages.appendChild(messageDiv);

        scrollToBottom();
        console.log('✅ Welcome message displayed');
    }

    function handleButtonClick(button) {
        console.log('\n' + '='.repeat(70));
        console.log('🔘 BUTTON CLICKED:', button.text);
        console.log('   Type:', button.type);
        console.log('   Value:', button.value);
        console.log('   Has Submenu:', button.has_submenu);
        console.log('='.repeat(70));

        if (button.has_submenu && button.submenu_items && button.submenu_items.length > 0) {
            console.log('📂 Opening submenu with', button.submenu_items.length, 'items');
            showSubmenu(button.submenu_items, button.text);
        } else {
            console.log('⚡ Executing button action');
            executeButtonAction(button, button.text);
        }
    }

    function showSubmenu(submenuItems, parentButtonText) {
        console.log('\n' + '─'.repeat(70));
        console.log('📂 SHOWING SUBMENU');
        console.log('   Parent:', parentButtonText);
        console.log('   Items:', submenuItems.length);
        console.log('─'.repeat(70));

        addMessage(`Selected: ${parentButtonText}`, 'user');

        const messageDiv = document.createElement('div');
        messageDiv.className = 'message bot-message submenu-message';

        const avatar = createBotAvatar();
        messageDiv.appendChild(avatar);

        const wrapper = document.createElement('div');
        wrapper.className = 'message-wrapper';

        const content = document.createElement('div');
        content.className = 'message-content';
        content.textContent = `Please choose from ${parentButtonText}:`;
        // ✅ FIX: Use config colors
        content.style.background = config.botMessageColor;
        content.style.color = config.botTextColor;
        wrapper.appendChild(content);

        const buttonsContainer = document.createElement('div');
        buttonsContainer.className = 'welcome-buttons submenu-buttons';

        const backBtn = document.createElement('button');
        backBtn.className = 'welcome-button back-button';
        backBtn.innerHTML = '← Back to Main Menu';
        backBtn.style.borderColor = '#64748b';
        backBtn.style.color = '#64748b';
        backBtn.style.background = 'white';
        backBtn.style.fontWeight = '600';

        backBtn.addEventListener('mouseenter', function() {
            this.style.background = '#64748b';
            this.style.color = 'white';
        });
        backBtn.addEventListener('mouseleave', function() {
            this.style.background = 'white';
            this.style.color = '#64748b';
        });

        backBtn.addEventListener('click', function() {
            console.log('⬅️ Back button clicked - returning to main menu');
            addMessage('Back to Main Menu', 'user');
            displayWelcomeMessage();
            currentSubmenuParent = null;
        });

        buttonsContainer.appendChild(backBtn);

        submenuItems.forEach((item, index) => {
            console.log(`   ${index + 1}. ${item.text} (${item.type})`);

            const btn = document.createElement('button');
            btn.className = 'welcome-button submenu-item-button';
            btn.textContent = item.text || 'Option';
            btn.style.borderColor = config.themeColor;
            btn.style.color = config.themeColor;
            btn.style.background = 'white';

            btn.addEventListener('mouseenter', function() {
                this.style.background = config.themeColor;
                this.style.color = 'white';
            });
            btn.addEventListener('mouseleave', function() {
                this.style.background = 'white';
                this.style.color = config.themeColor;
            });

            btn.addEventListener('click', function() {
                console.log('🎯 Submenu item clicked:', item.text);
                executeButtonAction(item, item.text);
                currentSubmenuParent = null;
            });

            buttonsContainer.appendChild(btn);
        });

        wrapper.appendChild(buttonsContainer);
        messageDiv.appendChild(wrapper);
        chatMessages.appendChild(messageDiv);

        scrollToBottom();
        console.log('✅ Submenu displayed with back button');

        currentSubmenuParent = parentButtonText;
    }

    function executeButtonAction(item, displayText) {
        const type = item.type || 'message';
        const value = item.value || item.text;
        const intentName = item.intent_name || null;

        console.log('\n' + '─'.repeat(70));
        console.log('⚡ EXECUTING ACTION');
        console.log('   Display Text:', displayText);
        console.log('   Action Type:', type);
        console.log('   Action Value:', value);
        console.log('─'.repeat(70));

        switch(type) {
            case 'url':
                console.log('🌐 Opening URL:', value);
                if (value && value.trim() !== '') {
                    addMessage(displayText, 'user');
                    showTypingIndicator();
                    setTimeout(() => {
                        removeTypingIndicator();
                        addMessage(`Opening: ${value}`, 'bot');
                        window.open(value, '_blank');
                    }, 500);
                } else {
                    console.warn('⚠️ No URL provided');
                }
                break;

            case 'message':
                console.log('💬 MESSAGE ACTION: Returning exact message text');

                addMessage(displayText, 'user');

                showTypingIndicator();
                setTimeout(() => {
                    removeTypingIndicator();
                    sendButtonActionToBot({
                        type: 'message',
                        value: value,
                        text: displayText
                    });
                }, 500);
                break;

            case 'intent':
                console.log('🎯 INTENT ACTION: Triggering intent');

                addMessage(displayText, 'user');

                showTypingIndicator();
                setTimeout(() => {
                    removeTypingIndicator();
                    sendButtonActionToBot({
                        type: 'intent',
                        intent_name: intentName || value,
                        value: value,
                        text: displayText
                    });
                }, 500);
                break;

            default:
                console.log('📤 Default action - sending as message');
                addMessage(displayText, 'user');
                showTypingIndicator();
                setTimeout(() => {
                    removeTypingIndicator();
                    sendMessageToBot(displayText);
                }, 500);
        }
    }

    function sendMessage() {
        const message = userInput.value.trim();
        if (!message) return;

        console.log('📤 Sending user input:', message);

        addMessage(message, 'user');
        userInput.value = '';
        sendBtn.disabled = true;

        showTypingIndicator();
        sendMessageToBot(message);
    }

    function sendButtonActionToBot(buttonAction) {
        console.log('\n' + '='.repeat(70));
        console.log('📡 API REQUEST - BUTTON ACTION');
        console.log('   Type:', buttonAction.type);
        console.log('   Value:', buttonAction.value);
        console.log('   Session ID:', sessionId || 'New session');
        console.log('='.repeat(70));

        const payload = {
            message: '',
            button_action: buttonAction
        };

        if (sessionId) {
            payload.session_id = sessionId;
        }

        fetch(`/api/chat/${config.embedCode}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        })
        .then(response => {
            console.log('📥 Response status:', response.status);
            if (!response.ok) {
                throw new Error(`Server error: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log('✅ Response data:', data);

            if (data.session_id && !sessionId) {
                sessionId = data.session_id;
                console.log('🔑 Session ID stored:', sessionId);
            }

            removeTypingIndicator();
            addMessage(data.response, 'bot');

            if (sendBtn) {
                sendBtn.disabled = false;
            }
            if (userInput) {
                userInput.focus();
            }
        })
        .catch(error => {
            console.error('❌ API Error:', error);
            removeTypingIndicator();
            addMessage('Sorry, I encountered an error. Please try again.', 'bot');

            if (sendBtn) {
                sendBtn.disabled = false;
            }
            if (userInput) {
                userInput.focus();
            }
        });
    }

    function sendMessageToBot(message, buttonAction = null) {
        console.log('\n' + '='.repeat(70));
        console.log('📡 API REQUEST - REGULAR MESSAGE');
        console.log('   Message:', message);
        console.log('   Session ID:', sessionId || 'New session');
        console.log('='.repeat(70));

        const payload = {
            message: message
        };

        if (sessionId) {
            payload.session_id = sessionId;
        }

        if (buttonAction) {
            payload.button_action = buttonAction;
        }

        fetch(`/api/chat/${config.embedCode}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        })
        .then(response => {
            console.log('📥 Response status:', response.status);
            if (!response.ok) {
                throw new Error(`Server error: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log('✅ Response data:', data);

            if (data.session_id && !sessionId) {
                sessionId = data.session_id;
                console.log('🔑 Session ID stored:', sessionId);
            }

            removeTypingIndicator();
            addMessage(data.response, 'bot');

            if (sendBtn) {
                sendBtn.disabled = false;
            }
            if (userInput) {
                userInput.focus();
            }
        })
        .catch(error => {
            console.error('❌ API Error:', error);
            removeTypingIndicator();
            addMessage('Sorry, I encountered an error. Please try again.', 'bot');

            if (sendBtn) {
                sendBtn.disabled = false;
            }
            if (userInput) {
                userInput.focus();
            }
        });
    }

    // ✅ CRITICAL FIX: Use config.botTextColor for bot messages
    function addMessage(text, sender) {
        console.log(`💬 Adding ${sender} message:`, text.substring(0, 50));

        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;

        if (sender === 'bot') {
            const avatar = createBotAvatar();
            messageDiv.appendChild(avatar);

            const wrapper = document.createElement('div');
            wrapper.className = 'message-wrapper';

            const content = document.createElement('div');
            content.className = 'message-content';
            content.textContent = text;

            // ✅ FIX: Use config colors instead of hardcoded values
            content.style.background = config.botMessageColor;
            content.style.color = config.botTextColor;

            console.log(`   Bot message colors: bg=${config.botMessageColor}, text=${config.botTextColor}`);

            wrapper.appendChild(content);
            messageDiv.appendChild(wrapper);
        } else {
            const content = document.createElement('div');
            content.className = 'message-content';
            content.textContent = text;

            // ✅ FIX: Use config.userTextColor
            content.style.cssText = `background: ${config.themeColor} !important; color: ${config.userTextColor} !important;`;

            messageDiv.appendChild(content);
        }

        chatMessages.appendChild(messageDiv);
        scrollToBottom();
    }

    function showTypingIndicator() {
        const typingDiv = document.createElement('div');
        typingDiv.className = 'message bot-message';
        typingDiv.id = 'typingIndicator';

        const avatar = createBotAvatar();
        typingDiv.appendChild(avatar);

        const wrapper = document.createElement('div');
        wrapper.className = 'message-wrapper';

        const indicator = document.createElement('div');
        indicator.className = 'typing-indicator';
        indicator.innerHTML = '<div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div>';

        wrapper.appendChild(indicator);
        typingDiv.appendChild(wrapper);

        chatMessages.appendChild(typingDiv);
        scrollToBottom();
    }

    function removeTypingIndicator() {
        const indicator = document.getElementById('typingIndicator');
        if (indicator) {
            indicator.remove();
        }
    }

    function scrollToBottom() {
        chatMessages.scrollTo({
            top: chatMessages.scrollHeight,
            behavior: 'smooth'
        });
    }

    if (userInput) {
        userInput.focus();
    }

    console.log('\n' + '='.repeat(70));
    console.log('✅ PREVIEW CHATBOT READY!');
    console.log('   Buttons loaded:', config.welcomeButtons.length);
    console.log('   Theme color:', config.themeColor);
    console.log('   Bot text color:', config.botTextColor);
    console.log('   User text color:', config.userTextColor);
    console.log('='.repeat(70) + '\n');
});

// ========== SIDEBAR MENUS ==========
const botsMenu = document.getElementById('botsMenu');
const botsSubmenu = document.getElementById('botsSubmenu');
const accountMenu = document.getElementById('accountMenu');
const accountSubmenu = document.getElementById('accountSubmenu');

if (botsMenu && botsSubmenu) {
    botsMenu.addEventListener('click', function(e) {
        e.preventDefault();
        if (accountSubmenu && accountSubmenu.classList.contains('open')) {
            accountSubmenu.classList.remove('open');
            accountMenu.classList.remove('expanded');
        }
        this.classList.toggle('expanded');
        botsSubmenu.classList.toggle('open');
    });
}

if (accountMenu && accountSubmenu) {
    accountMenu.addEventListener('click', function(e) {
        e.preventDefault();
        if (botsSubmenu && botsSubmenu.classList.contains('open')) {
            botsSubmenu.classList.remove('open');
            botsMenu.classList.remove('expanded');
        }
        this.classList.toggle('expanded');
        accountSubmenu.classList.toggle('open');
    });
}

document.addEventListener('click', function(e) {
    if (botsMenu && botsSubmenu && !e.target.closest('#botsMenu') && !e.target.closest('#botsSubmenu')) {
        botsSubmenu.classList.remove('open');
        botsMenu.classList.remove('expanded');
    }
    if (accountMenu && accountSubmenu && !e.target.closest('#accountMenu') && !e.target.closest('#accountSubmenu')) {
        accountSubmenu.classList.remove('open');
        accountMenu.classList.remove('expanded');
    }
});

// ========== MOBILE MENU ==========
const mobileMenuToggle = document.getElementById('mobileMenuToggle');
const sidebar = document.getElementById('sidebar');

if (mobileMenuToggle && sidebar) {
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
}

// ===== DYNAMIC CSS FOR SUBMENU STYLING =====
const style = document.createElement('style');
style.textContent = `
    .submenu-message {
        animation: slideIn 0.3s ease-out;
    }

    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(-10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .welcome-button.has-submenu {
        position: relative;
        padding-right: 30px;
    }

    .submenu-arrow {
        font-size: 10px;
        margin-left: 5px;
        transition: transform 0.3s ease;
    }

    .welcome-button.back-button {
        width: 100%;
        margin-bottom: 12px;
        background: #f8fafc !important;
        border-color: #64748b !important;
        color: #64748b !important;
        font-weight: 600;
    }

    .welcome-button.back-button:hover {
        background: #64748b !important;
        color: white !important;
    }

    .submenu-item-button {
        transition: all 0.2s ease;
    }

    .submenu-buttons {
        border-top: 1px solid #e2e8f0;
        padding-top: 12px;
        margin-top: 8px;
    }
`;
document.head.appendChild(style);