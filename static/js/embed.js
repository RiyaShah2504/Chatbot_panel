/**
 * ============================================================================
 * ✅ EMBED WIDGET - COMPLETE FIXED VERSION
 * ============================================================================
 * - Fixed avatar display (no theme overlay)
 * - Fixed badge rendering (shows "1" not "\")
 * - Same behavior as preview.js
 * ============================================================================
 */

const config = window.chatConfig;

console.log('🤖 Embed Widget Initialized');
console.log('Config:', config);

// Parse welcome buttons if needed
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
        </svg>
    `;
}

function createBotAvatar() {
    const avatar = document.createElement('div');
    avatar.className = 'msg-avatar';

    if (config.botAvatar) {
        const img = document.createElement('img');
        img.src = config.botAvatar;
        img.alt = config.botName;
        avatar.appendChild(img);
    } else {
        avatar.innerHTML = getRobotIconSVG(config.themeColor);
    }

    return avatar;
}

// ===== MAIN APPLICATION =====
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 Embed Widget Ready');

    const launcher = document.getElementById('launcher');
    const widget = document.getElementById('widget');
    const closeBtn = document.getElementById('closeBtn');
    const helpMessage = document.getElementById('helpMessage');
    const messagesContainer = document.getElementById('messages');
    const input = document.getElementById('input');
    const sendBtn = document.getElementById('sendBtn');

    let sessionId = null;
    let isOpen = false;

    // Apply theme colors
    document.documentElement.style.setProperty('--primary', config.themeColor);
    document.documentElement.style.setProperty('--bg-chat', config.chatBackgroundColor);
    document.documentElement.style.setProperty('--bot-bg', config.botMessageColor);
    document.documentElement.style.setProperty('--bot-text', config.botTextColor);
    document.documentElement.style.setProperty('--user-bg', config.themeColor);
    document.documentElement.style.setProperty('--user-text', config.userTextColor);

    // ✅ Hide help message after 5 seconds
    let helpTimeout;
    function startHelpTimeout() {
        helpTimeout = setTimeout(() => {
            if (helpMessage && !isOpen) {
                helpMessage.classList.add('hidden');
            }
        }, 10000);
    }

    function clearHelpTimeout() {
        if (helpTimeout) clearTimeout(helpTimeout);
    }

    startHelpTimeout();

    helpMessage.addEventListener('mouseenter', clearHelpTimeout);
    helpMessage.addEventListener('mouseleave', startHelpTimeout);


    // Launcher click
    launcher.addEventListener('click', function() {
        isOpen = !isOpen;

        if (isOpen) {
            widget.classList.add('open');
            launcher.classList.add('open');
            helpMessage.classList.add('hidden');

            // Show welcome message on first open
            if (messagesContainer.children.length === 0) {
                displayWelcomeMessage();
            }

            input.focus();
        } else {
            widget.classList.remove('open');
            launcher.classList.remove('open');
        }
    });


    // Close button
    closeBtn.addEventListener('click', function() {
        widget.classList.remove('open');
        launcher.classList.remove('open');
        isOpen = false;
    });

    // Send button
    sendBtn.addEventListener('click', function(e) {
        e.preventDefault();
        sendMessage();
    });

    // Enter key
    input.addEventListener('keypress', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // ===== WELCOME MESSAGE =====
    function displayWelcomeMessage() {
        console.log('💬 Displaying welcome message...');

        const messageDiv = document.createElement('div');
        messageDiv.className = 'message bot';

        const avatar = createBotAvatar();
        messageDiv.appendChild(avatar);

        const content = document.createElement('div');
        content.className = 'msg-content';

        const bubble = document.createElement('div');
        bubble.className = 'msg-bubble';
        bubble.textContent = config.welcomeMessage;
        content.appendChild(bubble);

        // Add buttons if available
        if (config.welcomeButtons && config.welcomeButtons.length > 0) {
            const buttonsContainer = document.createElement('div');
            buttonsContainer.className = 'buttons';

            config.welcomeButtons.forEach(button => {
                const btn = document.createElement('button');
                btn.className = 'btn';
                btn.textContent = button.text || 'Button';

                if (button.has_submenu && button.submenu_items && button.submenu_items.length > 0) {
                    btn.classList.add('has-submenu');
                    const arrow = document.createElement('span');
                    arrow.textContent = ' ▼';
                    btn.appendChild(arrow);
                }

                btn.addEventListener('click', () => handleButtonClick(button));
                buttonsContainer.appendChild(btn);
            });

            content.appendChild(buttonsContainer);
        }

        messageDiv.appendChild(content);
        messagesContainer.appendChild(messageDiv);
        scrollToBottom();

        console.log('✅ Welcome message displayed');
    }

    // ===== BUTTON HANDLER =====
    function handleButtonClick(button) {
        console.log('🔘 Button clicked:', button.text);

        if (button.has_submenu && button.submenu_items && button.submenu_items.length > 0) {
            // ✅ NO user message - just show submenu
            showSubmenu(button.submenu_items, button.text);
        } else {
            executeButtonAction(button, button.text);
        }
    }

    // ===== SUBMENU =====
    function showSubmenu(submenuItems, parentButtonText) {
        console.log('📂 Showing submenu for:', parentButtonText);

        const messageDiv = document.createElement('div');
        messageDiv.className = 'message bot';

        const avatar = createBotAvatar();
        messageDiv.appendChild(avatar);

        const content = document.createElement('div');
        content.className = 'msg-content';

        const bubble = document.createElement('div');
        bubble.className = 'msg-bubble';
        bubble.textContent = `Please choose from ${parentButtonText}:`;
        content.appendChild(bubble);

        const buttonsContainer = document.createElement('div');
        buttonsContainer.className = 'buttons';

        // ✅ Back to Main Menu button
        const backBtn = document.createElement('button');
        backBtn.className = 'btn back-button';
        backBtn.innerHTML = '← Back to Main Menu';
        backBtn.addEventListener('click', function() {
            console.log('⬅️ Back to Main Menu');
            // ✅ Just append welcome message (DO NOT clear previous chats)
            displayWelcomeMessage();
        });
        buttonsContainer.appendChild(backBtn);

        // Submenu items
        submenuItems.forEach(item => {
            const btn = document.createElement('button');
            btn.className = 'btn';
            btn.textContent = item.text || 'Option';
            btn.addEventListener('click', () => executeButtonAction(item, item.text));
            buttonsContainer.appendChild(btn);
        });

        content.appendChild(buttonsContainer);
        messageDiv.appendChild(content);
        messagesContainer.appendChild(messageDiv);
        scrollToBottom();
    }

    // ===== EXECUTE ACTION =====
    function executeButtonAction(item, displayText) {
        const type = item.type || 'message';
        const value = item.value || item.text;

        console.log('⚡ Executing:', type, value);

        switch(type) {
            case 'url':
                addMessage(displayText, 'user');
                showTypingIndicator();
                setTimeout(() => {
                    removeTypingIndicator();
                    addMessage(`Opening: ${value}`, 'bot');
                    window.open(value, '_blank');
                }, 500);
                break;

            case 'message':
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
                addMessage(displayText, 'user');
                showTypingIndicator();
                setTimeout(() => {
                    removeTypingIndicator();
                    sendButtonActionToBot({
                        type: 'intent',
                        value: value,
                        text: displayText
                    });
                }, 500);
                break;

            default:
                addMessage(displayText, 'user');
                showTypingIndicator();
                setTimeout(() => {
                    removeTypingIndicator();
                    sendMessageToBot(displayText);
                }, 500);
        }
    }

    // ===== SEND MESSAGE =====
    function sendMessage() {
        const message = input.value.trim();
        if (!message) return;

        addMessage(message, 'user');
        input.value = '';
        sendBtn.disabled = true;

        showTypingIndicator();
        sendMessageToBot(message);
    }

    // ===== API CALLS =====
    function sendButtonActionToBot(buttonAction) {
        const payload = { message: '', button_action: buttonAction };
        if (sessionId) payload.session_id = sessionId;

        fetch(`/api/chat/${config.embedCode}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        })
        .then(res => res.json())
        .then(data => {
            if (data.session_id && !sessionId) sessionId = data.session_id;
            removeTypingIndicator();
            addMessage(data.response, 'bot');
            sendBtn.disabled = false;
            input.focus();
        })
        .catch(error => {
            console.error('API Error:', error);
            removeTypingIndicator();
            addMessage('Sorry, I encountered an error. Please try again.', 'bot');
            sendBtn.disabled = false;
        });
    }

    function sendMessageToBot(message) {
        const payload = { message: message };
        if (sessionId) payload.session_id = sessionId;

        fetch(`/api/chat/${config.embedCode}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        })
        .then(res => res.json())
        .then(data => {
            if (data.session_id && !sessionId) sessionId = data.session_id;
            removeTypingIndicator();
            addMessage(data.response, 'bot');

            sendBtn.disabled = false;
            input.focus();
        })
        .catch(error => {
            console.error('API Error:', error);
            removeTypingIndicator();
            addMessage('Sorry, I encountered an error. Please try again.', 'bot');
            sendBtn.disabled = false;
        });
    }

    // ===== CONVERT URLs TO CLICKABLE LINKS =====
    function convertUrlsToLinks(text) {
        // URL regex pattern - matches http://, https://, and www. URLs
        const urlPattern = /(https?:\/\/[^\s<>"]+)|(www\.[^\s<>"]+)/g;

        return text.replace(urlPattern, function(url) {
            let href = url;
            let displayUrl = url;

            // Remove trailing punctuation
            const trailingPunct = url.match(/[.,;:!?)]$/);
            if (trailingPunct) {
                displayUrl = url.slice(0, -1);
                href = displayUrl;
            }

            // Add https:// if URL starts with www.
            if (displayUrl.startsWith('www.')) {
                href = 'https://' + displayUrl;
            }

            const link = `<a href="${href}" target="_blank" rel="noopener noreferrer" style="color: var(--primary, #4F46E5); text-decoration: underline; font-weight: 500;">${displayUrl}</a>${trailingPunct ? trailingPunct[0] : ''}`;
            return link;
        });
    }

    // ===== ADD MESSAGE =====
    function addMessage(text, sender) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}`;

        if (sender === 'bot') {
            const avatar = createBotAvatar();
            messageDiv.appendChild(avatar);

            const content = document.createElement('div');
            content.className = 'msg-content';

            const bubble = document.createElement('div');
            bubble.className = 'msg-bubble';

            // ✅ Convert URLs to clickable links for bot messages
            const processedText = convertUrlsToLinks(text);
            bubble.innerHTML = processedText;

            content.appendChild(bubble);
            messageDiv.appendChild(content);
        } else {
            // User messages - keep as plain text
            const bubble = document.createElement('div');
            bubble.className = 'msg-bubble';
            bubble.textContent = text;
            messageDiv.appendChild(bubble);
        }

        messagesContainer.appendChild(messageDiv);
        scrollToBottom();
    }

    // ===== TYPING INDICATOR =====
    function showTypingIndicator() {
        const typingDiv = document.createElement('div');
        typingDiv.className = 'message bot';
        typingDiv.id = 'typingIndicator';

        const avatar = createBotAvatar();
        typingDiv.appendChild(avatar);

        const content = document.createElement('div');
        content.className = 'msg-content';

        const indicator = document.createElement('div');
        indicator.className = 'typing-indicator';
        indicator.innerHTML = '<div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div>';
        content.appendChild(indicator);

        typingDiv.appendChild(content);
        messagesContainer.appendChild(typingDiv);
        scrollToBottom();
    }

    function removeTypingIndicator() {
        const indicator = document.getElementById('typingIndicator');
        if (indicator) indicator.remove();
    }

    function scrollToBottom() {
        messagesContainer.scrollTo({
            top: messagesContainer.scrollHeight,
            behavior: 'smooth'
        });
    }

    console.log('✅ Embed Widget Fully Initialized');
});