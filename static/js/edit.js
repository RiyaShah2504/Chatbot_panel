// ========== FORM ELEMENTS ==========
const themeColorInput = document.getElementById('theme_color');
const colorValue = document.querySelector('.color-value');
const welcomeMessageInput = document.getElementById('welcome_message');
const botNameInput = document.getElementById('bot_name');
const avatarInput = document.getElementById('bot_avatar');
const avatarPreview = document.getElementById('avatarPreview');
const removeAvatarBtn = document.getElementById('removeAvatar');
const adjustAvatarBtn = document.getElementById('adjustAvatarBtn');
const removeAvatarFlag = document.getElementById('remove_avatar_flag');

// Color inputs
const chatBgColor = document.getElementById('chat_background_color');
const botMsgColor = document.getElementById('bot_message_color');
const botTextColor = document.getElementById('bot_text_color');
const userTextColor = document.getElementById('user_text_color');

// Color value displays
const colorValueBg = document.querySelector('.color-value-bg');
const colorValueBotBg = document.querySelector('.color-value-bot-bg');
const colorValueBotText = document.querySelector('.color-value-bot-text');
const colorValueUserText = document.querySelector('.color-value-user-text');

// Preview elements
const previewHeader = document.getElementById('previewHeader');
const previewIcon = document.getElementById('previewIcon');
const previewBotName = document.getElementById('previewBotName');
const previewWelcome = document.getElementById('previewWelcome');
const previewAvatar = document.getElementById('previewAvatar');
const previewSendBtn = document.getElementById('previewSendBtn');
const previewBody = document.getElementById('previewBody');
const messageAvatar = document.querySelector('.message-avatar');
const userMessageContent = document.querySelector('.preview-message.user .message-content');
const botMessageContent = document.querySelector('.preview-message.bot .message-content');

// Welcome buttons
const addWelcomeButtonBtn = document.getElementById('addWelcomeButton');
const welcomeButtonsList = document.getElementById('welcomeButtonsList');
const previewButtonsContainer = document.getElementById('previewButtonsContainer');
const welcomeButtonsDataInput = document.getElementById('welcome_buttons');
const botAvatarDataInput = document.getElementById('bot_avatar_data');

// Avatar editor
const avatarEditorModal = document.getElementById('avatarEditorModal');
const zoomSlider = document.getElementById('zoomSlider');
const xSlider = document.getElementById('xSlider');
const ySlider = document.getElementById('ySlider');
const zoomValue = document.getElementById('zoomValue');
const xValue = document.getElementById('xValue');
const yValue = document.getElementById('yValue');
const previewCanvas = document.getElementById('previewCanvas');

// ========== GLOBAL STATE ==========
let uploadedAvatarUrl = null;
let originalAvatarUrl = null;
let welcomeButtons = [];
let buttonIdCounter = 0;
let submenuIdCounter = 0;
let currentX = 0;
let currentY = 0;
let currentZoom = 100;
let isDragging = false;
let dragStartX = 0;
let dragStartY = 0;

const CIRCLE_SIZE = 180;
const PREVIEW_SIZE = 100;

// Get existing avatar from page load
const existingAvatarImg = avatarPreview.querySelector('img');
if (existingAvatarImg) {
    uploadedAvatarUrl = existingAvatarImg.src;
    originalAvatarUrl = existingAvatarImg.src;
    console.log('Found existing avatar from database:', uploadedAvatarUrl);
}

// ========== HELPER FUNCTIONS ==========
function getValueLabel(type) {
    switch(type) {
        case 'url': return 'URL *';
        case 'intent': return 'Intent Name *';
        case 'message': return 'Message Text *';
        default: return 'Value *';
    }
}

function getValuePlaceholder(type) {
    switch(type) {
        case 'url': return 'https://example.com';
        case 'intent': return 'e.g., greeting, pricing';
        case 'message': return 'e.g., I want to contact support';
        default: return 'Enter value';
    }
}

function getModernRobotIconSVG(color) {
    return `
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" width="100%" height="100%">
            <rect x="16" y="24" width="32" height="28" rx="6" fill="${color}"/>
            <line x1="32" y1="16" x2="32" y2="24" stroke="${color}" stroke-width="2.5" stroke-linecap="round"/>
            <circle cx="32" cy="14" r="3" fill="${color}"/>
            <circle cx="24" cy="34" r="3.5" fill="white"/>
            <circle cx="40" cy="34" r="3.5" fill="white"/>
            <path d="M 24 44 Q 32 48 40 44" stroke="white" stroke-width="2.5" fill="none" stroke-linecap="round"/>
            <rect x="10" y="30" width="6" height="14" rx="3" fill="${color}" opacity="0.8"/>
            <rect x="48" y="30" width="6" height="14" rx="3" fill="${color}" opacity="0.8"/>
            <circle cx="24" cy="46" r="1.5" fill="white" opacity="0.6"/>
            <circle cx="32" cy="46" r="1.5" fill="white" opacity="0.6"/>
            <circle cx="40" cy="46" r="1.5" fill="white" opacity="0.6"/>
        </svg>
    `;
}

function getUserProfileIconSVG(color) {
    return `
        <svg width="40" height="40" viewBox="0 0 24 24" fill="${color}" stroke="none">
            <path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z"/>
        </svg>
    `;
}

function isValidUrl(string) {
    try {
        const url = new URL(string);
        return url.protocol === 'http:' || url.protocol === 'https:';
    } catch (_) {
        return false;
    }
}

// ========== AVATAR FUNCTIONS ==========
function updatePreviewAvatars() {
    const themeColor = themeColorInput?.value || '#4F46E5';
    const file = avatarInput?.files?.[0];

    const headerAvatar = document.getElementById('previewIcon');
    const messageAvatar = document.getElementById('previewAvatar');

    if (uploadedAvatarUrl) {
        const img = `<img src="${uploadedAvatarUrl}" alt="Avatar" style="width: 100%; height: 100%; object-fit: cover; border-radius: 50%;">`;
        headerAvatar.innerHTML = img;
        messageAvatar.innerHTML = img;
        headerAvatar.style.background = 'white';
        messageAvatar.style.background = 'white';
        messageAvatar.style.border = '2px solid #e2e8f0';
    } else if (file) {
        const reader = new FileReader();
        reader.onload = function (e) {
            const img = `<img src="${e.target.result}" alt="Avatar" style="width: 100%; height: 100%; object-fit: cover; border-radius: 50%;">`;
            headerAvatar.innerHTML = img;
            messageAvatar.innerHTML = img;
            headerAvatar.style.background = 'white';
            messageAvatar.style.background = 'white';
            messageAvatar.style.border = '2px solid #e2e8f0';
        };
        reader.readAsDataURL(file);
    } else {
        headerAvatar.innerHTML = getModernRobotIconSVG(themeColor);
        messageAvatar.innerHTML = getModernRobotIconSVG(themeColor);
        headerAvatar.style.background = 'white';
        headerAvatar.style.border = `2px solid ${themeColor}`;
        messageAvatar.style.background = 'white';
        messageAvatar.style.border = `2px solid ${themeColor}`;
    }
}

function updateUploadAvatar() {
    if (!uploadedAvatarUrl) {
        const staticColor = "#CBD5E1";
        avatarPreview.innerHTML = getUserProfileIconSVG(staticColor);
        avatarPreview.style.color = staticColor;
        avatarPreview.style.border = "2px dashed #e2e8f0";
    }
}

updatePreviewAvatars();
updateUploadAvatar();

// ========== LOAD EXISTING BUTTONS WITH SUBMENUS ==========
function loadExistingButtons() {
    const existingButtonsData = welcomeButtonsDataInput.value;

    if (!existingButtonsData || existingButtonsData === '[]' || existingButtonsData === '') {
        return;
    }

    try {
        const parsedButtons = JSON.parse(existingButtonsData);

        if (!Array.isArray(parsedButtons) || parsedButtons.length === 0) {
            return;
        }

        parsedButtons.forEach((button) => {
            const buttonId = buttonIdCounter++;
            const buttonType = button.type || 'url';
            const buttonText = button.text || '';
            const buttonValue = button.value || '';
            const hasSubmenu = button.has_submenu || false;
            const submenuItems = button.submenu_items || [];

            const buttonData = {
                id: buttonId,
                text: buttonText,
                type: buttonType,
                value: buttonValue,
                has_submenu: hasSubmenu,
                submenu_items: []
            };

            // Add submenu items
            if (hasSubmenu && Array.isArray(submenuItems)) {
                submenuItems.forEach((submenuItem) => {
                    const submenuId = submenuIdCounter++;
                    buttonData.submenu_items.push({
                        id: submenuId,
                        text: submenuItem.text || '',
                        type: submenuItem.type || 'url',
                        value: submenuItem.value || ''
                    });
                });
            }

            welcomeButtons.push(buttonData);
            renderButton(buttonData);
        });

        updatePreviewButtons();

    } catch (e) {
        console.error('Error parsing existing buttons:', e);
    }
}

// ========== RENDER BUTTON ==========
function renderButton(buttonData) {
    const buttonItem = document.createElement('div');
    buttonItem.className = 'button-item' + (buttonData.has_submenu ? ' has-submenu' : '');
    buttonItem.dataset.buttonId = buttonData.id;

    const submenuBadge = buttonData.has_submenu ?
        `<span class="submenu-badge">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <polyline points="9 18 15 12 9 6"></polyline>
            </svg>
            ${buttonData.submenu_items.length} items
        </span>` : '';

    buttonItem.innerHTML = `
        <div class="button-header">
            <span class="button-number">
                Button ${welcomeButtons.findIndex(b => b.id === buttonData.id) + 1}
                ${submenuBadge}
            </span>
            <button type="button" class="btn-remove-button" onclick="removeWelcomeButton(${buttonData.id})">Remove</button>
        </div>
        <div class="button-fields">
            <div>
                <label>Button Text *</label>
                <input type="text" class="button-text" value="${buttonData.text}" placeholder="e.g., Our Services" required data-button-id="${buttonData.id}">
            </div>
            <div class="field-row">
                <div>
                    <label>Action Type *</label>
                    <select class="button-type" data-button-id="${buttonData.id}">
                        <option value="url" ${buttonData.type === 'url' ? 'selected' : ''}>Open URL</option>
                        <option value="intent" ${buttonData.type === 'intent' ? 'selected' : ''}>Trigger Intent</option>
                        <option value="message" ${buttonData.type === 'message' ? 'selected' : ''}>Send Message</option>
                    </select>
                </div>
                <div class="button-value-container">
                    <label class="button-value-label">${getValueLabel(buttonData.type)}</label>
                    <input type="text" class="button-value" value="${buttonData.value}" placeholder="${getValuePlaceholder(buttonData.type)}" data-button-id="${buttonData.id}">
                </div>
            </div>

            <!-- Submenu Toggle -->
            <div class="submenu-toggle-section">
                <label>
                    <input type="checkbox" class="submenu-checkbox" data-button-id="${buttonData.id}" ${buttonData.has_submenu ? 'checked' : ''}>
                </label>
                <span class="submenu-toggle-text">Add Submenu/Sub Services</span>
            </div>

            <!-- Submenu Items Container -->
            <div class="submenu-items-container ${buttonData.has_submenu ? 'visible' : ''}" id="submenu-container-${buttonData.id}">
                <div class="submenu-header">
                    <span class="submenu-title">📋 Submenu Items</span>
                    <div class="submenu-header-actions">
                        <button type="button" class="btn-add-submenu" onclick="addSubmenuItem(${buttonData.id})">
                            <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <line x1="12" y1="5" x2="12" y2="19"></line>
                                <line x1="5" y1="12" x2="19" y2="12"></line>
                            </svg>
                            Add Item
                        </button>
                        <button type="button" class="btn-add-submenu btn-remove-submenu-header" onclick="clearAllSubmenu(${buttonData.id})">
                            Remove All
                        </button>
                    </div>
                </div>
                <div class="submenu-items-list" id="submenu-list-${buttonData.id}">
                    ${buttonData.submenu_items.length === 0 ? '<div class="submenu-empty-state">No submenu items yet. Click "Add Item" below.</div>' : ''}
                </div>
                <div class="submenu-info-box">
                    <strong>💡 Tip:</strong> Use submenus to organize services (e.g., "Web Design", "SEO", "Marketing") under a main category.
                </div>
            </div>
        </div>
    `;

    welcomeButtonsList.appendChild(buttonItem);

    // Add event listeners
    const textInput = buttonItem.querySelector('.button-text');
    const typeSelect = buttonItem.querySelector('.button-type');
    const valueInput = buttonItem.querySelector('.button-value');
    const valueLabel = buttonItem.querySelector('.button-value-label');
    const submenuCheckbox = buttonItem.querySelector('.submenu-checkbox');

    textInput.addEventListener('input', function() {
        updateButtonData(buttonData.id, 'text', this.value);
        updatePreviewButtons();
    });

    typeSelect.addEventListener('change', function() {
        updateButtonData(buttonData.id, 'type', this.value);
        valueLabel.textContent = getValueLabel(this.value);
        valueInput.placeholder = getValuePlaceholder(this.value);
        updatePreviewButtons();
    });

    valueInput.addEventListener('input', function() {
        updateButtonData(buttonData.id, 'value', this.value);
        updatePreviewButtons();
    });

    submenuCheckbox.addEventListener('change', function() {
        toggleSubmenu(buttonData.id, this.checked);
    });

    // Render existing submenu items
    if (buttonData.has_submenu && buttonData.submenu_items.length > 0) {
        buttonData.submenu_items.forEach((submenuItem) => {
            renderSubmenuItem(buttonData.id, submenuItem);
        });
    }
}

// ========== TOGGLE SUBMENU ==========
function toggleSubmenu(buttonId, isChecked) {
    const button = welcomeButtons.find(b => b.id === buttonId);
    if (button) {
        button.has_submenu = isChecked;

        const container = document.getElementById(`submenu-container-${buttonId}`);
        const buttonItem = document.querySelector(`.button-item[data-button-id="${buttonId}"]`);

        if (container) {
            if (isChecked) {
                container.classList.add('visible');
                buttonItem.classList.add('has-submenu');
            } else {
                container.classList.remove('visible');
                buttonItem.classList.remove('has-submenu');
            }
        }

        updateHiddenInput();
        updatePreviewButtons();
    }
}

window.toggleSubmenu = toggleSubmenu;

// ========== ADD SUBMENU ITEM ==========
window.addSubmenuItem = function(buttonId) {
    const button = welcomeButtons.find(b => b.id === buttonId);
    if (!button) return;

    if (button.submenu_items.length >= 10) {
        alert('Maximum 10 submenu items allowed per button');
        return;
    }

    const submenuId = submenuIdCounter++;
    const submenuItem = {
        id: submenuId,
        text: '',
        type: 'url',
        value: ''
    };

    button.submenu_items.push(submenuItem);
    renderSubmenuItem(buttonId, submenuItem);

    // Remove empty state if exists
    const emptyState = document.querySelector(`#submenu-list-${buttonId} .submenu-empty-state`);
    if (emptyState) emptyState.remove();

    updateHiddenInput();
    updatePreviewButtons();
};

// ========== RENDER SUBMENU ITEM - FIXED WITH REMOVE BUTTON ==========
function renderSubmenuItem(buttonId, submenuItem) {
    const submenuList = document.getElementById(`submenu-list-${buttonId}`);
    if (!submenuList) return;

    const button = welcomeButtons.find(b => b.id === buttonId);
    const itemIndex = button.submenu_items.findIndex(item => item.id === submenuItem.id);

    const submenuItemEl = document.createElement('div');
    submenuItemEl.className = 'submenu-item';
    submenuItemEl.dataset.submenuId = submenuItem.id;

    submenuItemEl.innerHTML = `
        <div class="submenu-item-header">
            <span class="submenu-item-number">Sub-item ${itemIndex + 1}</span>
            <button type="button" class="btn-remove-submenu" onclick="removeSubmenuItem(${buttonId}, ${submenuItem.id})">Remove</button>
        </div>
        <div class="submenu-item-fields">
            <div>
                <label>Text *</label>
                <input type="text" class="submenu-text" value="${submenuItem.text}" placeholder="e.g., Web Design" data-submenu-id="${submenuItem.id}">
            </div>
            <div class="field-row">
                <div>
                    <label>Type *</label>
                    <select class="submenu-type" data-submenu-id="${submenuItem.id}">
                        <option value="url" ${submenuItem.type === 'url' ? 'selected' : ''}>URL</option>
                        <option value="intent" ${submenuItem.type === 'intent' ? 'selected' : ''}>Intent</option>
                        <option value="message" ${submenuItem.type === 'message' ? 'selected' : ''}>Message</option>
                    </select>
                </div>
                <div>
                    <label class="submenu-value-label">${getValueLabel(submenuItem.type)}</label>
                    <input type="text" class="submenu-value" value="${submenuItem.value}" placeholder="${getValuePlaceholder(submenuItem.type)}" data-submenu-id="${submenuItem.id}">
                </div>
            </div>
        </div>
    `;

    submenuList.appendChild(submenuItemEl);

    // Add event listeners
    const textInput = submenuItemEl.querySelector('.submenu-text');
    const typeSelect = submenuItemEl.querySelector('.submenu-type');
    const valueInput = submenuItemEl.querySelector('.submenu-value');
    const valueLabel = submenuItemEl.querySelector('.submenu-value-label');

    textInput.addEventListener('input', function() {
        updateSubmenuData(buttonId, submenuItem.id, 'text', this.value);
        updatePreviewButtons();
    });

    typeSelect.addEventListener('change', function() {
        updateSubmenuData(buttonId, submenuItem.id, 'type', this.value);
        valueLabel.textContent = getValueLabel(this.value);
        valueInput.placeholder = getValuePlaceholder(this.value);
        updatePreviewButtons();
    });

    valueInput.addEventListener('input', function() {
        updateSubmenuData(buttonId, submenuItem.id, 'value', this.value);
        updatePreviewButtons();
    });
}

// ========== REMOVE SUBMENU ITEM ==========
window.removeSubmenuItem = function(buttonId, submenuId) {
    const button = welcomeButtons.find(b => b.id === buttonId);
    if (!button) return;

    const itemIndex = button.submenu_items.findIndex(item => item.id === submenuId);
    if (itemIndex !== -1) {
        button.submenu_items.splice(itemIndex, 1);

        const submenuItemEl = document.querySelector(`.submenu-item[data-submenu-id="${submenuId}"]`);
        if (submenuItemEl) submenuItemEl.remove();

        // Update numbering
        const submenuList = document.getElementById(`submenu-list-${buttonId}`);
        submenuList.querySelectorAll('.submenu-item-number').forEach((el, index) => {
            el.textContent = `Sub-item ${index + 1}`;
        });

        // Show empty state if no items
        if (button.submenu_items.length === 0) {
            submenuList.innerHTML = '<div class="submenu-empty-state">No submenu items yet. Click "Add Item" below.</div>';
        }

        updateHiddenInput();
        updatePreviewButtons();
    }
};

// ========== CLEAR ALL SUBMENU ==========
window.clearAllSubmenu = function(buttonId) {
    const button = welcomeButtons.find(b => b.id === buttonId);
    if (!button) return;

    if (button.submenu_items.length === 0) {
        alert('No submenu items to remove');
        return;
    }

    if (confirm(`Are you sure you want to remove all ${button.submenu_items.length} submenu items?`)) {
        button.submenu_items = [];

        const submenuList = document.getElementById(`submenu-list-${buttonId}`);
        if (submenuList) {
            submenuList.innerHTML = '<div class="submenu-empty-state">No submenu items yet. Click "Add Item" below.</div>';
        }

        updateHiddenInput();
        updatePreviewButtons();
    }
};

// ========== UPDATE SUBMENU DATA ==========
function updateSubmenuData(buttonId, submenuId, field, value) {
    const button = welcomeButtons.find(b => b.id === buttonId);
    if (button) {
        const submenuItem = button.submenu_items.find(item => item.id === submenuId);
        if (submenuItem) {
            submenuItem[field] = value;
            updateHiddenInput();
        }
    }
}

// ========== ADD WELCOME BUTTON ==========
function addWelcomeButton() {
    if (welcomeButtons.length >= 10) {
        alert('Maximum 10 buttons allowed');
        return;
    }

    const buttonId = buttonIdCounter++;
    const buttonData = {
        id: buttonId,
        text: '',
        type: 'url',
        value: '',
        has_submenu: false,
        submenu_items: []
    };

    welcomeButtons.push(buttonData);
    renderButton(buttonData);
    updateHiddenInput();
    updatePreviewButtons();
}

if (addWelcomeButtonBtn) {
    addWelcomeButtonBtn.addEventListener('click', addWelcomeButton);
}

// ========== REMOVE WELCOME BUTTON ==========
window.removeWelcomeButton = function(buttonId) {
    const buttonIndex = welcomeButtons.findIndex(b => b.id === buttonId);
    if (buttonIndex !== -1) {
        welcomeButtons.splice(buttonIndex, 1);

        const buttonItem = document.querySelector(`.button-item[data-button-id="${buttonId}"]`);
        if (buttonItem) buttonItem.remove();

        // Update numbering
        document.querySelectorAll('.button-number').forEach((el, index) => {
            const submenuBadge = el.querySelector('.submenu-badge');
            const badgeHTML = submenuBadge ? submenuBadge.outerHTML : '';
            el.innerHTML = `Button ${index + 1} ${badgeHTML}`;
        });

        updatePreviewButtons();
        updateHiddenInput();
    }
};

// ========== UPDATE BUTTON DATA ==========
function updateButtonData(buttonId, field, value) {
    const button = welcomeButtons.find(b => b.id === buttonId);
    if (button) {
        button[field] = value;
        updateHiddenInput();
    }
}

// ========== UPDATE HIDDEN INPUT ==========
function updateHiddenInput() {
    const buttonData = welcomeButtons
        .filter(b => b.text.trim() !== '')
        .map(b => {
            const data = {
                text: b.text,
                type: b.type,
                value: b.value,
                has_submenu: b.has_submenu
            };

            if (b.has_submenu) {
                data.submenu_items = b.submenu_items
                    .filter(item => item.text.trim() !== '')
                    .map(item => ({
                        text: item.text,
                        type: item.type,
                        value: item.value
                    }));
            } else {
                data.submenu_items = [];
            }

            return data;
        });

    welcomeButtonsDataInput.value = JSON.stringify(buttonData);
    console.log('Updated buttons data:', buttonData);
}

// ========== UPDATE PREVIEW BUTTONS WITH SUBMENU - FIXED VERSION ==========
function updatePreviewButtons() {
    if (previewButtonsContainer) {
        previewButtonsContainer.innerHTML = '';
    }

    const validButtons = welcomeButtons.filter(b => b.text.trim() !== '');

    if (validButtons.length === 0) return;

    const themeColor = themeColorInput ? themeColorInput.value : '#4F46E5';

    validButtons.forEach((button) => {
        const wrapper = document.createElement('div');
        wrapper.className = 'preview-button-wrapper';

        const previewBtn = document.createElement('button');
        previewBtn.className = 'preview-button' + (button.has_submenu ? ' has-submenu' : '');
        previewBtn.textContent = button.text;
        previewBtn.style.borderColor = themeColor;
        previewBtn.style.color = themeColor;
        previewBtn.style.background = 'white';
        previewBtn.type = 'button';

        // Add submenu dropdown if has submenu
        if (button.has_submenu && button.submenu_items.length > 0) {
            const dropdown = document.createElement('div');
            dropdown.className = 'preview-submenu-dropdown';

            button.submenu_items.forEach((submenuItem) => {
                const submenuBtn = document.createElement('button');
                submenuBtn.className = 'preview-submenu-item';
                submenuBtn.textContent = submenuItem.text;
                submenuBtn.style.color = themeColor;
                submenuBtn.type = 'button';

                submenuBtn.addEventListener('mouseenter', function() {
                    this.style.background = '#f7fafc';
                });

                submenuBtn.addEventListener('mouseleave', function() {
                    this.style.background = 'white';
                });

                submenuBtn.addEventListener('click', function(e) {
                    e.stopPropagation();
                    if (submenuItem.type === 'message' && submenuItem.value.trim()) {
                        showMessagePreview(submenuItem.value);
                    }
                    dropdown.classList.remove('active');
                });

                dropdown.appendChild(submenuBtn);
            });

            wrapper.appendChild(dropdown);

            // Toggle dropdown on main button click
            previewBtn.addEventListener('click', function(e) {
                e.stopPropagation();

                // Close other dropdowns
                document.querySelectorAll('.preview-submenu-dropdown.active').forEach(d => {
                    if (d !== dropdown) d.classList.remove('active');
                });

                dropdown.classList.toggle('active');
            });

            // Hover effect for submenu button
            previewBtn.addEventListener('mouseenter', function() {
                this.style.background = '#f7fafc';
            });

            previewBtn.addEventListener('mouseleave', function() {
                this.style.background = 'white';
            });

        } else {
            // Hover effects for non-submenu buttons
            previewBtn.addEventListener('mouseenter', function() {
                this.style.background = themeColor;
                this.style.color = 'white';
            });

            previewBtn.addEventListener('mouseleave', function() {
                this.style.background = 'white';
                this.style.color = themeColor;
            });

            // Regular message button
            if (button.type === 'message' && button.value.trim()) {
                previewBtn.addEventListener('click', function() {
                    showMessagePreview(button.value);
                });
            }
        }

        wrapper.appendChild(previewBtn);

        if (previewButtonsContainer) {
            previewButtonsContainer.appendChild(wrapper);
        }
    });

    // Close dropdowns when clicking outside
    document.addEventListener('click', function(e) {
        if (!e.target.closest('.preview-button-wrapper')) {
            document.querySelectorAll('.preview-submenu-dropdown.active').forEach(d => {
                d.classList.remove('active');
            });
        }
    });
}

function showMessagePreview(messageText) {
    const existingPreview = previewBody.querySelector('.message-preview-demo');
    if (existingPreview) existingPreview.remove();

    const userMsg = document.createElement('div');
    userMsg.className = 'preview-message user message-preview-demo';
    userMsg.innerHTML = `<div class="message-content user-message">${messageText}</div>`;

    const botMsg = document.createElement('div');
    botMsg.className = 'preview-message bot message-preview-demo';

    const themeColor = themeColorInput?.value || '#4F46E5';
    const avatarContent = uploadedAvatarUrl
        ? `<img src="${uploadedAvatarUrl}" alt="Avatar" style="width: 100%; height: 100%; object-fit: cover; border-radius: 50%;">`
        : getModernRobotIconSVG(themeColor);

    botMsg.innerHTML = `
        <div class="message-avatar" style="background: white; border: 2px solid ${uploadedAvatarUrl ? '#e2e8f0' : themeColor};">${avatarContent}</div>
        <div class="message-wrapper">
            <div class="message-content bot-message">Thank you for your message: "${messageText}". Our team will respond shortly!</div>
        </div>
    `;

    previewBody.appendChild(userMsg);
    previewBody.appendChild(botMsg);
    previewBody.scrollTop = previewBody.scrollHeight;

    setTimeout(() => {
        userMsg.remove();
        botMsg.remove();
    }, 5000);
}

// ========== LOAD EXISTING BUTTONS ON PAGE LOAD ==========
loadExistingButtons();

// ========== THEME COLOR UPDATE ==========
if (themeColorInput && colorValue) {
    themeColorInput.addEventListener('input', function() {
        const color = this.value;
        colorValue.textContent = color;
        previewHeader.style.background = color;
        previewSendBtn.style.background = color;
        if (userMessageContent) {
            userMessageContent.style.background = color;
        }
        updatePreviewButtons();
        updatePreviewAvatars();
        updateUploadAvatar();
    });
}

// ========== COLOR UPDATES ==========
if (chatBgColor && colorValueBg) {
    chatBgColor.addEventListener('input', function() {
        const color = this.value;
        colorValueBg.textContent = color;
        previewBody.style.background = color;
    });
}

if (botMsgColor && colorValueBotBg) {
    botMsgColor.addEventListener('input', function() {
        const color = this.value;
        colorValueBotBg.textContent = color;
        if (botMessageContent) {
            botMessageContent.style.background = color;
        }
    });
}

if (botTextColor && colorValueBotText) {
    botTextColor.addEventListener('input', function() {
        const color = this.value;
        colorValueBotText.textContent = color;
        if (botMessageContent) {
            botMessageContent.style.color = color;
        }
    });
}

if (userTextColor && colorValueUserText) {
    userTextColor.addEventListener('input', function() {
        const color = this.value;
        colorValueUserText.textContent = color;
        if (userMessageContent) {
            userMessageContent.style.color = color;
        }
    });
}

// ========== WELCOME MESSAGE ==========
welcomeMessageInput.addEventListener('input', function() {
    previewWelcome.textContent = this.value || 'Hello! How can I help you?';
});

// ========== BOT NAME ==========
botNameInput.addEventListener('input', function() {
    const name = this.value || 'AI Assistant';
    previewBotName.textContent = name;
});

// ========== AVATAR UPLOAD ==========
avatarInput.addEventListener('change', function(e) {
    const file = e.target.files[0];

    if (file) {
        if (file.size > 2 * 1024 * 1024) {
            alert('File size must be less than 2MB');
            this.value = '';
            return;
        }

        const allowedTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/gif', 'image/svg+xml'];
        if (!allowedTypes.includes(file.type)) {
            alert('Please upload a valid image file (PNG, JPG, GIF, or SVG)');
            this.value = '';
            return;
        }

        const reader = new FileReader();
        reader.onload = function(event) {
            uploadedAvatarUrl = event.target.result;
            botAvatarDataInput.value = event.target.result;
            originalAvatarUrl = event.target.result;

            avatarPreview.innerHTML = `<img src="${uploadedAvatarUrl}" alt="Avatar" style="width: 100%; height: 100%; object-fit: cover; border-radius: 50%;">`;
            updatePreviewAvatars();

            removeAvatarBtn.style.display = 'inline-flex';
            adjustAvatarBtn.style.display = 'inline-flex';
            if (removeAvatarFlag) {
                removeAvatarFlag.value = 'false';
            }

            currentZoom = 100;
            currentX = 0;
            currentY = 0;
        };
        reader.readAsDataURL(file);
    }
});

// ========== REMOVE AVATAR ==========
if (removeAvatarBtn) {
    removeAvatarBtn.addEventListener('click', function() {
        uploadedAvatarUrl = null;
        originalAvatarUrl = null;
        avatarInput.value = '';

        updateUploadAvatar();
        updatePreviewAvatars();

        this.style.display = 'none';
        adjustAvatarBtn.style.display = 'none';

        if (removeAvatarFlag) {
            removeAvatarFlag.value = 'true';
        }
        if (botAvatarDataInput) {
            botAvatarDataInput.value = '';
        }
    });
}

// ========== AVATAR EDITOR ==========
window.openAvatarEditor = function() {
    if (!uploadedAvatarUrl) return;
    avatarEditorModal.classList.add('open');
    renderAvatarPreview();
};

window.closeAvatarEditor = function() {
    avatarEditorModal.classList.remove('open');
};

function updateAvatarSliderValues() {
    zoomValue.textContent = zoomSlider.value;
    xValue.textContent = xSlider.value + 'px';
    yValue.textContent = ySlider.value + 'px';
}

function renderAvatarPreview() {
    const ctx = previewCanvas.getContext('2d');
    const img = new Image();

    img.onload = function() {
        previewCanvas.width = CIRCLE_SIZE;
        previewCanvas.height = CIRCLE_SIZE;

        ctx.clearRect(0, 0, CIRCLE_SIZE, CIRCLE_SIZE);

        ctx.beginPath();
        ctx.arc(CIRCLE_SIZE / 2, CIRCLE_SIZE / 2, CIRCLE_SIZE / 2, 0, Math.PI * 2);
        ctx.clip();

        const scale = currentZoom / 100;
        const scaledWidth = img.width * scale;
        const scaledHeight = img.height * scale;

        const x = (CIRCLE_SIZE - scaledWidth) / 2 + currentX;
        const y = (CIRCLE_SIZE - scaledHeight) / 2 + currentY;

        ctx.drawImage(img, x, y, scaledWidth, scaledHeight);
    };

    img.src = originalAvatarUrl || uploadedAvatarUrl;
}

if (zoomSlider) {
    zoomSlider.addEventListener('input', function() {
        currentZoom = parseInt(this.value);
        updateAvatarSliderValues();
        renderAvatarPreview();
    });
}

if (xSlider) {
    xSlider.addEventListener('input', function() {
        currentX = parseInt(this.value);
        updateAvatarSliderValues();
        renderAvatarPreview();
    });
}

if (ySlider) {
    ySlider.addEventListener('input', function() {
        currentY = parseInt(this.value);
        updateAvatarSliderValues();
        renderAvatarPreview();
    });
}

if (previewCanvas) {
    previewCanvas.addEventListener('mousedown', function(e) {
        isDragging = true;
        dragStartX = e.clientX;
        dragStartY = e.clientY;
    });

    document.addEventListener('mousemove', function(e) {
        if (!isDragging) return;

        const deltaX = (e.clientX - dragStartX) / 2;
        const deltaY = (e.clientY - dragStartY) / 2;

        xSlider.value = Math.max(-150, Math.min(150, currentX + deltaX));
        ySlider.value = Math.max(-150, Math.min(150, currentY + deltaY));

        currentX = parseInt(xSlider.value);
        currentY = parseInt(ySlider.value);

        updateAvatarSliderValues();
        renderAvatarPreview();

        dragStartX = e.clientX;
        dragStartY = e.clientY;
    });

    document.addEventListener('mouseup', function() {
        isDragging = false;
    });
}

window.resetAvatarImage = function() {
    currentX = 0;
    currentY = 0;
    currentZoom = 100;

    zoomSlider.value = 100;
    xSlider.value = 0;
    ySlider.value = 0;

    updateAvatarSliderValues();
    renderAvatarPreview();
};

// ========== SAVE AVATAR IMAGE ==========
window.saveAvatarImage = function() {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    const img = new Image();

    img.onload = function() {
        canvas.width = PREVIEW_SIZE;
        canvas.height = PREVIEW_SIZE;

        ctx.clearRect(0, 0, PREVIEW_SIZE, PREVIEW_SIZE);

        ctx.beginPath();
        ctx.arc(PREVIEW_SIZE / 2, PREVIEW_SIZE / 2, PREVIEW_SIZE / 2, 0, Math.PI * 2);
        ctx.clip();

        const scale = currentZoom / 100;
        const scaledWidth = img.width * scale;
        const scaledHeight = img.height * scale;

        const x = (PREVIEW_SIZE - scaledWidth) / 2 + (currentX * PREVIEW_SIZE / CIRCLE_SIZE);
        const y = (PREVIEW_SIZE - scaledHeight) / 2 + (currentY * PREVIEW_SIZE / CIRCLE_SIZE);

        ctx.drawImage(img, x, y, scaledWidth, scaledHeight);

        uploadedAvatarUrl = canvas.toDataURL('image/png');

        if (botAvatarDataInput) {
            botAvatarDataInput.value = uploadedAvatarUrl;
        }

        avatarPreview.innerHTML = `<img src="${uploadedAvatarUrl}" alt="Avatar" style="width: 100%; height: 100%; object-fit: cover; border-radius: 50%;">`;
        updatePreviewAvatars();

        closeAvatarEditor();
    };

    img.onerror = function() {
        alert('Error processing image. Please try again.');
    };

    img.src = originalAvatarUrl || uploadedAvatarUrl;
};

if (avatarEditorModal) {
    avatarEditorModal.addEventListener('click', function(e) {
        if (e.target === avatarEditorModal) {
            closeAvatarEditor();
        }
    });
}

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

// ========== FORM VALIDATION ==========
const form = document.getElementById('editChatbotForm');
if (form) {
    form.addEventListener('submit', function(e) {
        const name = document.getElementById('name').value.trim();
        const botName = document.getElementById('bot_name').value.trim();
        const welcomeMessage = document.getElementById('welcome_message').value.trim();

        if (!name) {
            e.preventDefault();
            alert('Please enter a chatbot name');
            return;
        }

        if (!botName) {
            e.preventDefault();
            alert('Please enter a bot display name');
            return;
        }

        if (!welcomeMessage) {
            e.preventDefault();
            alert('Please enter a welcome message');
            return;
        }

        const buttons = JSON.parse(welcomeButtonsDataInput.value || '[]');
        for (let i = 0; i < buttons.length; i++) {
            const button = buttons[i];
            if (!button.text.trim()) {
                e.preventDefault();
                alert(`Button ${i + 1}: Please enter button text`);
                return;
            }

            // Validate main button value if not a submenu
            if (!button.has_submenu && !button.value.trim()) {
                e.preventDefault();
                let fieldName = 'value';
                if (button.type === 'url') fieldName = 'URL';
                else if (button.type === 'intent') fieldName = 'Intent name';
                else if (button.type === 'message') fieldName = 'Message text';
                alert(`Button ${i + 1}: Please enter ${fieldName}`);
                return;
            }

            if (!button.has_submenu && button.type === 'url' && !isValidUrl(button.value)) {
                e.preventDefault();
                alert(`Button ${i + 1}: Please enter a valid URL (starting with http:// or https://)`);
                return;
            }

            // Validate submenu items if has submenu
            if (button.has_submenu && button.submenu_items) {
                for (let j = 0; j < button.submenu_items.length; j++) {
                    const submenuItem = button.submenu_items[j];

                    if (!submenuItem.text.trim()) {
                        e.preventDefault();
                        alert(`Button ${i + 1} - Submenu item ${j + 1}: Please enter text`);
                        return;
                    }

                    if (!submenuItem.value.trim()) {
                        e.preventDefault();
                        let fieldName = 'value';
                        if (submenuItem.type === 'url') fieldName = 'URL';
                        else if (submenuItem.type === 'intent') fieldName = 'Intent name';
                        else if (submenuItem.type === 'message') fieldName = 'Message text';
                        alert(`Button ${i + 1} - Submenu item ${j + 1}: Please enter ${fieldName}`);
                        return;
                    }

                    if (submenuItem.type === 'url' && !isValidUrl(submenuItem.value)) {
                        e.preventDefault();
                        alert(`Button ${i + 1} - Submenu item ${j + 1}: Please enter a valid URL`);
                        return;
                    }
                }
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


