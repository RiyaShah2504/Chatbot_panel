// ================================================================
// train.js - COMPLETE & PERFECTED VERSION
// ================================================================

// ===== GLOBAL STATE =====
let currentPage = 'main'; // 'main' or 'upload'
let uploadMode = 'replace'; // 'replace' or 'append'
let allQAData = [];
let filteredQAData = [];
let currentFilter = 'all';
let currentSort = 'recent';
let searchQuery = '';

// File upload states
let selectedFile = null;
let selectedTextFile = null;
let selectedExcelFile = null;

// Manual Q&A states
let qaCount = 0;

// ===== INITIALIZE ON PAGE LOAD =====
document.addEventListener('DOMContentLoaded', function() {
    console.log('Training page initialized');
    initializeSidebar();
    initializeTabNavigation();
    initializeSearchAndFilter();
    initializeManualQA();

    // Check if we have training data
    const mainPage = document.getElementById('mainPage');
    if (mainPage && mainPage.querySelector('.main-page-header')) {
        loadQACards();
    }
});

// ===== SIDEBAR NAVIGATION =====
function initializeSidebar() {
    const mobileToggle = document.getElementById('mobileMenuToggle');
    const sidebar = document.getElementById('sidebar');

    if (mobileToggle && sidebar) {
        mobileToggle.addEventListener('click', () => {
            sidebar.classList.toggle('open');
        });

        // Close sidebar when clicking outside on mobile
        document.addEventListener('click', (e) => {
            if (window.innerWidth <= 968 &&
                sidebar.classList.contains('open') &&
                !sidebar.contains(e.target) &&
                !mobileToggle.contains(e.target)) {
                sidebar.classList.remove('open');
            }
        });
    }

    // Expandable menus
    const botsMenu = document.getElementById('botsMenu');
    const botsSubmenu = document.getElementById('botsSubmenu');
    if (botsMenu && botsSubmenu) {
        botsMenu.addEventListener('click', (e) => {
            e.preventDefault();
            botsMenu.classList.toggle('expanded');
            botsSubmenu.classList.toggle('open');
        });
    }

    const accountMenu = document.getElementById('accountMenu');
    const accountSubmenu = document.getElementById('accountSubmenu');
    if (accountMenu && accountSubmenu) {
        accountMenu.addEventListener('click', (e) => {
            e.preventDefault();
            accountMenu.classList.toggle('expanded');
            accountSubmenu.classList.toggle('open');
        });
    }
}

// ===== PAGE NAVIGATION =====
window.showUploadPage = function(mode) {
    uploadMode = mode;
    currentPage = 'upload';

    const mainPage = document.getElementById('mainPage');
    const uploadPage = document.getElementById('uploadPage');
    const uploadPageTitle = document.getElementById('uploadPageTitle');
    const uploadPageSubtitle = document.getElementById('uploadPageSubtitle');

    if (mainPage) mainPage.style.display = 'none';
    if (uploadPage) uploadPage.style.display = 'block';

    if (uploadPageTitle && uploadPageSubtitle) {
        if (mode === 'append') {
            uploadPageTitle.textContent = 'Add More Training Data';
            uploadPageSubtitle.textContent = 'Add new Q&A pairs to existing training data';
        } else {
            uploadPageTitle.textContent = 'Replace Training Data';
            uploadPageSubtitle.textContent = 'Replace all existing training data with new data';
        }
    }

    window.scrollTo({ top: 0, behavior: 'smooth' });
};

window.backToMainPage = function() {
    currentPage = 'main';

    const mainPage = document.getElementById('mainPage');
    const uploadPage = document.getElementById('uploadPage');

    if (mainPage) mainPage.style.display = 'block';
    if (uploadPage) uploadPage.style.display = 'none';

    // Reset all upload forms
    resetAllUploadForms();

    window.scrollTo({ top: 0, behavior: 'smooth' });
};

function resetAllUploadForms() {
    // Reset JSON upload
    resetJsonUpload();
    // Reset text upload
    resetTextUpload();
    // Reset Excel upload
    resetExcelUpload();
}

// ===== TAB NAVIGATION =====
function initializeTabNavigation() {
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const tabName = btn.dataset.tab;

            // Update button states
            tabBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            // Update content visibility
            tabContents.forEach(content => {
                content.classList.remove('active');
                if (content.id === `${tabName}Tab`) {
                    content.classList.add('active');
                }
            });
        });
    });
}

// ===== SEARCH AND FILTER =====
function initializeSearchAndFilter() {
    const searchInput = document.getElementById('searchInput');
    const clearSearchBtn = document.getElementById('clearSearchBtn');
    const filterChips = document.querySelectorAll('.filter-chip');
    const sortSelect = document.getElementById('sortSelect');

    if (searchInput) {
        searchInput.addEventListener('input', (e) => {
            searchQuery = e.target.value.toLowerCase();
            if (clearSearchBtn) {
                clearSearchBtn.style.display = searchQuery ? 'block' : 'none';
            }
            filterAndRenderQA();
        });
    }

    if (clearSearchBtn) {
        clearSearchBtn.addEventListener('click', () => {
            if (searchInput) {
                searchInput.value = '';
                searchQuery = '';
                clearSearchBtn.style.display = 'none';
                filterAndRenderQA();
            }
        });
    }

    filterChips.forEach(chip => {
        chip.addEventListener('click', () => {
            currentFilter = chip.dataset.filter;
            filterChips.forEach(c => c.classList.remove('active'));
            chip.classList.add('active');
            filterAndRenderQA();
        });
    });

    if (sortSelect) {
        sortSelect.addEventListener('change', (e) => {
            currentSort = e.target.value;
            filterAndRenderQA();
        });
    }
}

function filterAndRenderQA() {
    let filtered = [...allQAData];

    // Apply search filter
    if (searchQuery) {
        filtered = filtered.filter(qa =>
            qa.question.toLowerCase().includes(searchQuery) ||
            qa.answer.toLowerCase().includes(searchQuery)
        );
    }

    // Apply time filter
    if (currentFilter === 'recent') {
        const sevenDaysAgo = new Date();
        sevenDaysAgo.setDate(sevenDaysAgo.getDate() - 7);
        filtered = filtered.filter(qa => {
            const qaDate = new Date(qa.created_at || qa.updated_at);
            return qaDate >= sevenDaysAgo;
        });
    }

    // Apply sorting
    filtered.sort((a, b) => {
        switch (currentSort) {
            case 'recent':
                return new Date(b.updated_at || b.created_at) - new Date(a.updated_at || a.created_at);
            case 'oldest':
                return new Date(a.created_at) - new Date(b.created_at);
            case 'az':
                return a.question.localeCompare(b.question);
            case 'za':
                return b.question.localeCompare(a.question);
            default:
                return 0;
        }
    });

    filteredQAData = filtered;
    renderQACards(filtered);
}

// ===== LOAD Q&A CARDS =====
async function loadQACards() {
    const qaCardsContainer = document.getElementById('qaCardsContainer');
    if (!qaCardsContainer) return;

    // Show loading state
    qaCardsContainer.innerHTML = `
        <div class="qa-cards-loading">
            <div class="loading-spinner"></div>
            <p>Loading Q&A pairs...</p>
        </div>
    `;

    try {
        const response = await fetch(`/chatbot/get-qa-pairs/${CHATBOT_ID}`);
        const result = await response.json();

        if (result.success && result.qa_pairs && result.qa_pairs.length > 0) {
            allQAData = result.qa_pairs;
            filteredQAData = result.qa_pairs;
            renderQACards(result.qa_pairs);

            // Update counts
            const resultsCount = document.getElementById('resultsCount');
            const totalCount = document.getElementById('totalCount');
            if (resultsCount) resultsCount.textContent = result.qa_pairs.length;
            if (totalCount) totalCount.textContent = result.qa_pairs.length;
        } else {
            showEmptyCardsState();
        }
    } catch (error) {
        console.error('Load error:', error);
        showEmptyCardsState();
    }
}

function renderQACards(qaPairs) {
    const qaCardsContainer = document.getElementById('qaCardsContainer');
    if (!qaCardsContainer) return;

    if (!qaPairs || qaPairs.length === 0) {
        showEmptyCardsState();
        return;
    }

    qaCardsContainer.innerHTML = qaPairs.map((qa, index) => {
        const createdDate = qa.created_at ? formatTimestamp(qa.created_at) : 'Unknown';
        const updatedDate = qa.updated_at ? formatTimestamp(qa.updated_at) : null;
        const wasUpdated = updatedDate && updatedDate !== createdDate && qa.updated_at !== qa.created_at;

        return `
            <div class="qa-card" data-qa-id="${qa.id || index}">
                <div class="qa-card-question">
                    <div class="qa-card-label">Question</div>
                    <div class="qa-card-text">${escapeHtml(qa.question)}</div>
                </div>
                <div class="qa-card-answer">
                    <div class="qa-card-label">Answer</div>
                    <div class="qa-card-text">${escapeHtml(qa.answer)}</div>
                </div>
                <div class="qa-card-meta">
                    ${wasUpdated ? `
                        <div class="qa-card-timestamp updated">
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <circle cx="12" cy="12" r="10"></circle>
                                <polyline points="12 6 12 12 16 14"></polyline>
                            </svg>
                            Updated: ${updatedDate}
                        </div>
                    ` : `
                        <div class="qa-card-timestamp">
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <circle cx="12" cy="12" r="10"></circle>
                                <polyline points="12 6 12 12 16 14"></polyline>
                            </svg>
                            ${createdDate}
                        </div>
                    `}
                    <div class="qa-card-actions">
                        <button class="btn-edit-card" onclick="editQACard(${qa.id || index})">
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"></path>
                                <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"></path>
                            </svg>
                            Edit
                        </button>
                        <button class="btn-delete-card" onclick="deleteQACard(${qa.id || index})">
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <polyline points="3 6 5 6 21 6"></polyline>
                                <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                            </svg>
                            Delete
                        </button>
                    </div>
                </div>
            </div>
        `;
    }).join('');

    // Update results count
    const resultsCount = document.getElementById('resultsCount');
    if (resultsCount) resultsCount.textContent = qaPairs.length;
}

function showEmptyCardsState() {
    const qaCardsContainer = document.getElementById('qaCardsContainer');
    if (!qaCardsContainer) return;

    qaCardsContainer.innerHTML = `
        <div class="qa-cards-empty">
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
            </svg>
            <h3>No matching Q&A pairs found</h3>
            <p>Try adjusting your search or filters</p>
        </div>
    `;

    const resultsCount = document.getElementById('resultsCount');
    if (resultsCount) resultsCount.textContent = '0';
}

// ===== EDIT Q&A MODAL =====
window.editQACard = function(qaId) {
    const qa = allQAData.find(q => q.id === qaId);
    if (!qa) {
        showAlert('Q&A pair not found', 'error');
        return;
    }

    console.log('📝 Opening edit modal for Q&A:', qaId);

    // Create modal overlay
    const overlay = document.createElement('div');
    overlay.className = 'edit-modal-overlay';
    overlay.id = 'editModalOverlay';

    // Create modal
    overlay.innerHTML = `
        <div class="edit-modal" onclick="event.stopPropagation()">
            <div class="edit-modal-header">
                <div class="edit-modal-title">
                    <span>Edit Q&A Pair</span>
                </div>
                <button class="btn-close-modal" onclick="closeEditModal()">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <line x1="18" y1="6" x2="6" y2="18"></line>
                        <line x1="6" y1="6" x2="18" y2="18"></line>
                    </svg>
                </button>
            </div>
            <div class="edit-modal-body">
                <div class="edit-form-group">
                    <label for="editQuestion">Question</label>
                    <textarea id="editQuestion" placeholder="Enter the question..." rows="3">${escapeHtml(qa.question)}</textarea>
                </div>
                <div class="edit-form-group">
                    <label for="editAnswer">Answer</label>
                    <textarea id="editAnswer" placeholder="Enter the answer..." rows="4">${escapeHtml(qa.answer)}</textarea>
                </div>
            </div>
            <div class="edit-modal-footer">
                <button class="btn-cancel-edit" onclick="closeEditModal()">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <line x1="18" y1="6" x2="6" y2="18"></line>
                        <line x1="6" y1="6" x2="18" y2="18"></line>
                    </svg>
                    Cancel
                </button>
                <button class="btn-save-edit" onclick="saveEditedQA(${qaId})">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <polyline points="20 6 9 17 4 12"></polyline>
                    </svg>
                    Save Changes
                </button>
            </div>
        </div>
    `;

    // Add to body
    document.body.appendChild(overlay);

    // Close on overlay click
    overlay.addEventListener('click', () => closeEditModal());

    // Auto-resize textareas
    const textareas = overlay.querySelectorAll('textarea');
    textareas.forEach(textarea => {
        textarea.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = this.scrollHeight + 'px';
        });
        // Trigger initial resize
        setTimeout(() => {
            textarea.style.height = 'auto';
            textarea.style.height = textarea.scrollHeight + 'px';
        }, 10);
    });

    // Focus on question field
    setTimeout(() => {
        document.getElementById('editQuestion')?.focus();
    }, 100);
};

// Close edit modal
window.closeEditModal = function() {
    const overlay = document.getElementById('editModalOverlay');
    if (overlay) {
        overlay.style.animation = 'fadeOut 0.2s ease';
        setTimeout(() => overlay.remove(), 200);
    }
};

// Save edited Q&A from modal
window.saveEditedQA = async function(qaId) {
    const questionInput = document.getElementById('editQuestion');
    const answerInput = document.getElementById('editAnswer');
    const saveBtn = document.querySelector('.btn-save-edit');

    if (!questionInput || !answerInput) return;

    const question = questionInput.value.trim();
    const answer = answerInput.value.trim();

    // Validation
    if (!question || !answer) {
        showAlert('Question and answer are required', 'error');
        if (!question) questionInput.classList.add('error');
        if (!answer) answerInput.classList.add('error');
        return;
    }

    // Remove error states
    questionInput.classList.remove('error');
    answerInput.classList.remove('error');

    // Disable button
    if (saveBtn) {
        saveBtn.disabled = true;
        saveBtn.innerHTML = '<span class="loading-spinner"></span><span>Saving...</span>';
    }

    console.log('💾 Saving edited Q&A:', qaId);

    try {
        const response = await fetch(`/chatbot/update-qa-pair/${qaId}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question, answer })
        });

        const result = await response.json();

        if (result.success) {
            showAlert('Q&A pair updated successfully!', 'success');
            closeEditModal();
            await loadQACards();
        } else {
            showAlert(result.message || 'Failed to update Q&A pair', 'error');
            // Re-enable button
            if (saveBtn) {
                saveBtn.disabled = false;
                saveBtn.innerHTML = `
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <polyline points="20 6 9 17 4 12"></polyline>
                    </svg>
                    Save Changes
                `;
            }
        }
    } catch (error) {
        console.error('Save edit error:', error);
        showAlert('An error occurred while saving', 'error');
        // Re-enable button
        if (saveBtn) {
            saveBtn.disabled = false;
            saveBtn.innerHTML = `
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <polyline points="20 6 9 17 4 12"></polyline>
                </svg>
                Save Changes
            `;
        }
    }
};

// Add fadeOut animation to CSS (add this CSS if not present)
const style = document.createElement('style');
style.textContent = `
    @keyframes fadeOut {
        from { opacity: 1; }
        to { opacity: 0; }
    }
`;
document.head.appendChild(style);

window.deleteQACard = async function(qaId) {
    if (!confirm('Are you sure you want to delete this Q&A pair? This will retrain your chatbot.')) return;

    try {
        const response = await fetch(`/chatbot/delete-qa-pair/${qaId}`, { method: 'POST' });
        const result = await response.json();

        if (result.success) {
            showAlert('Q&A pair deleted successfully', 'success');
            await loadQACards();
        } else {
            showAlert(result.message || 'Failed to delete Q&A pair', 'error');
        }
    } catch (error) {
        console.error('Delete error:', error);
        showAlert('An error occurred while deleting', 'error');
    }
};

// ===== DELETE ALL TRAINING DATA =====
window.deleteAllTrainingData = async function() {
    if (!confirm('⚠️ WARNING: This will delete ALL training data and reset your chatbot. Are you absolutely sure?')) {
        return;
    }

    try {
        const response = await fetch(`/chatbot/delete-training/${CHATBOT_ID}`, {
            method: 'POST'
        });

        const result = await response.json();

        if (result.success) {
            showAlert('All training data deleted successfully', 'success');
            setTimeout(() => window.location.reload(), 1500);
        } else {
            showAlert(result.message || 'Failed to delete training data', 'error');
        }
    } catch (error) {
        console.error('Delete error:', error);
        showAlert('An error occurred while deleting', 'error');
    }
};

// ===== DOWNLOAD INTENTS =====

window.downloadIntents = function () {
    try {
        // Let browser handle download so filename from Flask is preserved
        window.location.href = `/chatbot/download-intents/${CHATBOT_ID}`;

        showAlert('Downloading intents...', 'success');
    } catch (error) {
        console.error('Download error:', error);
        showAlert('Failed to download intents', 'error');
    }
};


// ===== DOWNLOAD SAMPLE =====
window.downloadSample = function(type) {
    let content, filename, mimeType;

    if (type === 'json') {
        content = JSON.stringify({
            "intents": [
                {
                    "tag": "greeting",
                    "patterns": ["Hi", "Hello", "Hey there", "Good morning"],
                    "responses": ["Hello! How can I help you today?", "Hi there! What can I do for you?"]
                },
                {
                    "tag": "hours",
                    "patterns": ["What are your hours?", "When are you open?", "Business hours"],
                    "responses": ["We're open Monday-Friday, 9 AM to 6 PM EST."]
                }
            ]
        }, null, 2);
        filename = 'sample_training_data.json';
        mimeType = 'application/json';
    } else if (type === 'text') {
        content = `Q: What are your business hours?
A: We are open Monday to Friday, 9 AM to 6 PM EST.

Q: How can I contact customer support?
A: You can reach our support team via email at support@example.com or call us at (555) 123-4567.`;
        filename = 'sample_training_data.txt';
        mimeType = 'text/plain';
    } else if (type === 'excel') {
        content = 'Question,Answer\n' +
                  '"What are your business hours?","We are open Monday to Friday, 9 AM to 6 PM EST."\n' +
                  '"How can I contact support?","You can reach us at support@example.com or call (555) 123-4567."\n' +
                  '"Do you offer refunds?","Yes, we offer a 30-day money-back guarantee."';
        filename = 'sample_training_data.csv';
        mimeType = 'text/csv';
    }

    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);

    showAlert(`Sample file "${filename}" downloaded successfully!`, 'success');
};

// ===== UTILITY FUNCTIONS =====
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatTimestamp(timestamp) {
    if (!timestamp) return 'Unknown';
    const date = new Date(timestamp);
    return date.toLocaleString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
}

function showAlert(message, type = 'info') {
    const alertContainer = document.getElementById('alertContainer');
    if (!alertContainer) return;

    const alert = document.createElement('div');
    alert.className = `alert alert-${type}`;
    alert.innerHTML = `
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            ${type === 'success' ? '<polyline points="20 6 9 17 4 12"></polyline>' :
              type === 'error' ? '<circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line>' :
              '<circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line>'}
        </svg>
        <span>${message}</span>
    `;

    alertContainer.appendChild(alert);

    setTimeout(() => {
        alert.style.opacity = '0';
        setTimeout(() => alert.remove(), 300);
    }, 5000);
}

// ===== JSON FILE UPLOAD =====
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const fileInfo = document.getElementById('fileInfo');
const fileName = document.getElementById('fileName');
const fileSize = document.getElementById('fileSize');
const removeFileBtn = document.getElementById('removeFile');
const uploadBtn = document.getElementById('uploadBtn');
const uploadForm = document.getElementById('uploadForm');
const progressBar = document.getElementById('progressBar');
const progressFill = document.getElementById('progressFill');

if (uploadArea && fileInput) {
    uploadArea.addEventListener('click', () => fileInput.click());

    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('dragover'));

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) handleFileSelect(e.dataTransfer.files[0]);
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) handleFileSelect(e.target.files[0]);
    });
}

function handleFileSelect(file) {
    if (!file || !file.name) return;
    if (!file.name.toLowerCase().endsWith('.json')) {
        showAlert('Please select a JSON file', 'error');
        return;
    }

    selectedFile = file;
    if (fileName) fileName.textContent = file.name;
    if (fileSize) fileSize.textContent = formatFileSize(file.size);

    if (uploadArea) uploadArea.style.display = 'none';
    if (fileInfo) fileInfo.style.display = 'flex';
    if (uploadBtn) uploadBtn.disabled = false;
}

if (removeFileBtn) {
    removeFileBtn.addEventListener('click', () => resetJsonUpload());
}

function resetJsonUpload() {
    selectedFile = null;
    if (fileInput) fileInput.value = '';
    if (fileInfo) fileInfo.style.display = 'none';
    if (uploadArea) uploadArea.style.display = 'flex';
    if (uploadBtn) uploadBtn.disabled = true;
    if (progressBar) progressBar.style.display = 'none';
    if (progressFill) progressFill.style.width = '0';
}

if (uploadForm) {
    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        if (!selectedFile) {
            showAlert('Please select a file', 'error');
            return;
        }

        const formData = new FormData();
        formData.append('training_file', selectedFile);
        formData.append('upload_mode', uploadMode);

        if (uploadBtn) {
            uploadBtn.disabled = true;
            uploadBtn.innerHTML = '<span class="loading-spinner"></span><span>Processing...</span>';
        }
        if (progressBar) progressBar.style.display = 'block';

        let progress = 0;
        const progressInterval = setInterval(() => {
            progress += 5;
            if (progress <= 90 && progressFill) progressFill.style.width = progress + '%';
        }, 100);

        try {
            const response = await fetch(`/chatbot/train/${CHATBOT_ID}`, {
                method: 'POST',
                body: formData
            });

            clearInterval(progressInterval);
            if (progressFill) progressFill.style.width = '100%';

            const result = await response.json();

            if (result.success) {
                showAlert(result.message, 'success');
                setTimeout(() => window.location.reload(), 1200);
            } else {
                showAlert(result.message || 'Failed to upload training file', 'error');
                resetUploadButton();
            }
        } catch (error) {
            console.error('Upload error:', error);
            clearInterval(progressInterval);
            showAlert('An error occurred during upload', 'error');
            resetUploadButton();
        }
    });
}

function resetUploadButton() {
    if (uploadBtn) {
        uploadBtn.disabled = false;
        uploadBtn.innerHTML = `
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <polyline points="20 6 9 17 4 12"></polyline>
            </svg>
            Upload & Train
        `;
    }
    if (progressBar) progressBar.style.display = 'none';
    if (progressFill) progressFill.style.width = '0';
}

// ===== TEXT FILE UPLOAD =====
const textUploadArea = document.getElementById('textUploadArea');
const textFileInput = document.getElementById('textFileInput');
const textFileInfo = document.getElementById('textFileInfo');
const textFileName = document.getElementById('textFileName');
const textFileSize = document.getElementById('textFileSize');
const removeTextFileBtn = document.getElementById('removeTextFile');
const textUploadBtn = document.getElementById('textUploadBtn');
const textUploadForm = document.getElementById('textUploadForm');
const textProgressBar = document.getElementById('textProgressBar');
const textProgressFill = document.getElementById('textProgressFill');

if (textUploadArea && textFileInput) {
    textUploadArea.addEventListener('click', () => textFileInput.click());

    textUploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        textUploadArea.classList.add('dragover');
    });

    textUploadArea.addEventListener('dragleave', () => textUploadArea.classList.remove('dragover'));

    textUploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        textUploadArea.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) handleTextFileSelect(e.dataTransfer.files[0]);
    });

    textFileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) handleTextFileSelect(e.target.files[0]);
    });
}

function handleTextFileSelect(file) {
    if (!file || !file.name) return;
    const validExtensions = ['.txt', '.html', '.htm'];
    const fileExtension = file.name.substring(file.name.lastIndexOf('.')).toLowerCase();

    if (!validExtensions.includes(fileExtension)) {
        showAlert('Please select a TXT or HTML file', 'error');
        return;
    }

    selectedTextFile = file;
    if (textFileName) textFileName.textContent = file.name;
    if (textFileSize) textFileSize.textContent = formatFileSize(file.size);

    if (textUploadArea) textUploadArea.style.display = 'none';
    if (textFileInfo) textFileInfo.style.display = 'flex';
    if (textUploadBtn) textUploadBtn.disabled = false;
}

if (removeTextFileBtn) {
    removeTextFileBtn.addEventListener('click', () => resetTextUpload());
}

function resetTextUpload() {
    selectedTextFile = null;
    if (textFileInput) textFileInput.value = '';
    if (textFileInfo) textFileInfo.style.display = 'none';
    if (textUploadArea) textUploadArea.style.display = 'flex';
    if (textUploadBtn) textUploadBtn.disabled = true;
    if (textProgressBar) textProgressBar.style.display = 'none';
    if (textProgressFill) textProgressFill.style.width = '0';
}

// ===== TEXT FILE UPLOAD FORM SUBMISSION =====
if (textUploadForm) {
    textUploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        if (!selectedTextFile) {
            showAlert('Please select a file', 'error');
            return;
        }

        const formData = new FormData();
        formData.append('text_file', selectedTextFile);
        formData.append('upload_mode', uploadMode);

        if (textUploadBtn) {
            textUploadBtn.disabled = true;
            textUploadBtn.innerHTML = '<span class="loading-spinner"></span><span>Processing...</span>';
        }
        if (textProgressBar) textProgressBar.style.display = 'block';

        let progress = 0;
        const progressInterval = setInterval(() => {
            progress += 5;
            if (progress <= 90 && textProgressFill) textProgressFill.style.width = progress + '%';
        }, 100);

        try {
            const response = await fetch(`/chatbot/train-text/${CHATBOT_ID}`, {
                method: 'POST',
                body: formData
            });

            clearInterval(progressInterval);
            if (textProgressFill) textProgressFill.style.width = '100%';

            const result = await response.json();

            if (result.success) {
                showAlert(result.message, 'success');
                setTimeout(() => window.location.reload(), 1200);
            } else {
                showAlert(result.message || 'Failed to upload text file', 'error');
                resetTextUploadButton();
            }
        } catch (error) {
            console.error('Text upload error:', error);
            clearInterval(progressInterval);
            showAlert('An error occurred during upload', 'error');
            resetTextUploadButton();
        }
    });
}

function resetTextUploadButton() {
    if (textUploadBtn) {
        textUploadBtn.disabled = false;
        textUploadBtn.innerHTML = `
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <polyline points="20 6 9 17 4 12"></polyline>
            </svg>
            Upload & Train
        `;
    }
    if (textProgressBar) textProgressBar.style.display = 'none';
    if (textProgressFill) textProgressFill.style.width = '0';
}

// ===== EXCEL FILE UPLOAD =====
const excelUploadArea = document.getElementById('excelUploadArea');
const excelFileInput = document.getElementById('excelFileInput');
const excelFileInfo = document.getElementById('excelFileInfo');
const excelFileName = document.getElementById('excelFileName');
const excelFileSize = document.getElementById('excelFileSize');
const removeExcelFileBtn = document.getElementById('removeExcelFile');
const excelUploadBtn = document.getElementById('excelUploadBtn');
const excelUploadForm = document.getElementById('excelUploadForm');
const excelProgressBar = document.getElementById('excelProgressBar');
const excelProgressFill = document.getElementById('excelProgressFill');

if (excelUploadArea && excelFileInput) {
    excelUploadArea.addEventListener('click', () => excelFileInput.click());

    excelUploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        excelUploadArea.classList.add('dragover');
    });

    excelUploadArea.addEventListener('dragleave', () => excelUploadArea.classList.remove('dragover'));

    excelUploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        excelUploadArea.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) handleExcelFileSelect(e.dataTransfer.files[0]);
    });

    excelFileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) handleExcelFileSelect(e.target.files[0]);
    });
}

function handleExcelFileSelect(file) {
    if (!file || !file.name) return;
    const validExtensions = ['.xlsx', '.xls', '.csv'];
    const fileExtension = file.name.substring(file.name.lastIndexOf('.')).toLowerCase();

    if (!validExtensions.includes(fileExtension)) {
        showAlert('Please select an Excel file (.xlsx, .xls, .csv)', 'error');
        return;
    }

    selectedExcelFile = file;
    if (excelFileName) excelFileName.textContent = file.name;
    if (excelFileSize) excelFileSize.textContent = formatFileSize(file.size);

    if (excelUploadArea) excelUploadArea.style.display = 'none';
    if (excelFileInfo) excelFileInfo.style.display = 'flex';
    if (excelUploadBtn) excelUploadBtn.disabled = false;
}

if (removeExcelFileBtn) {
    removeExcelFileBtn.addEventListener('click', () => resetExcelUpload());
}

function resetExcelUpload() {
    selectedExcelFile = null;
    if (excelFileInput) excelFileInput.value = '';
    if (excelFileInfo) excelFileInfo.style.display = 'none';
    if (excelUploadArea) excelUploadArea.style.display = 'flex';
    if (excelUploadBtn) excelUploadBtn.disabled = true;
    if (excelProgressBar) excelProgressBar.style.display = 'none';
    if (excelProgressFill) excelProgressFill.style.width = '0';
}

if (excelUploadForm) {
    excelUploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        if (!selectedExcelFile) {
            showAlert('Please select a file', 'error');
            return;
        }

        const formData = new FormData();
        formData.append('excel_file', selectedExcelFile);
        formData.append('upload_mode', uploadMode);

        if (excelUploadBtn) {
            excelUploadBtn.disabled = true;
            excelUploadBtn.innerHTML = '<span class="loading-spinner"></span><span>Processing...</span>';
        }
        if (excelProgressBar) excelProgressBar.style.display = 'block';

        let progress = 0;
        const progressInterval = setInterval(() => {
            progress += 5;
            if (progress <= 90 && excelProgressFill) excelProgressFill.style.width = progress + '%';
        }, 100);

        try {
            const response = await fetch(`/chatbot/train-excel/${CHATBOT_ID}`, {
                method: 'POST',
                body: formData
            });

            clearInterval(progressInterval);
            if (excelProgressFill) excelProgressFill.style.width = '100%';

            const result = await response.json();

            if (result.success) {
                showAlert(result.message, 'success');
                setTimeout(() => window.location.reload(), 1200);
            } else {
                showAlert(result.message || 'Failed to upload Excel file', 'error');
                resetExcelUploadButton();
            }
        } catch (error) {
            console.error('Excel upload error:', error);
            clearInterval(progressInterval);
            showAlert('An error occurred during upload', 'error');
            resetExcelUploadButton();
        }
    });
}

function resetExcelUploadButton() {
    if (excelUploadBtn) {
        excelUploadBtn.disabled = false;
        excelUploadBtn.innerHTML = `
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <polyline points="20 6 9 17 4 12"></polyline>
            </svg>
            Upload & Train
        `;
    }
    if (excelProgressBar) excelProgressBar.style.display = 'none';
    if (excelProgressFill) excelProgressFill.style.width = '0';
}

// ===== MANUAL Q&A ENTRY =====
function initializeManualQA() {
    const addQaBtn = document.getElementById('addQaBtn');
    const clearAllBtn = document.getElementById('clearAllBtn');
    const saveManualBtn = document.getElementById('saveManualBtn');
    const qaContainer = document.getElementById('qaContainer');

    if (addQaBtn) {
        addQaBtn.addEventListener('click', () => addQaPair());
    }

    if (clearAllBtn) {
        clearAllBtn.addEventListener('click', () => {
            if (confirm('Are you sure you want to clear all Q&A pairs?')) {
                if (qaContainer) qaContainer.innerHTML = '';
                qaCount = 0;
                showAlert('All Q&A pairs cleared', 'info');
            }
        });
    }

    if (saveManualBtn) {
        saveManualBtn.addEventListener('click', () => saveManualQA());
    }

    // Add initial Q&A pair
    if (qaContainer && qaContainer.children.length === 0) {
        addQaPair();
    }
}

function addQaPair(question = '', answer = '', qaId = null) {
    const qaContainer = document.getElementById('qaContainer');
    if (!qaContainer) return;

    qaCount++;
    const pairId = qaId || `qa_${qaCount}`;

    const qaCard = document.createElement('div');
    qaCard.className = 'qa-pair-card';
    qaCard.dataset.qaId = pairId;
    qaCard.innerHTML = `
        <div class="qa-pair-header">
            <span class="qa-pair-number">Q&A Pair #${qaCount}</span>
            <button type="button" class="btn-remove-qa" onclick="removeQaPair('${pairId}')">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <line x1="18" y1="6" x2="6" y2="18"></line>
                    <line x1="6" y1="6" x2="18" y2="18"></line>
                </svg>
            </button>
        </div>
        <div class="form-group">
            <label for="question_${pairId}">Question</label>
            <textarea
                id="question_${pairId}"
                class="form-input qa-question"
                rows="2"
                placeholder="Enter the question..."
                required
            >${question}</textarea>
        </div>
        <div class="form-group">
            <label for="answer_${pairId}">Answer</label>
            <textarea
                id="answer_${pairId}"
                class="form-input qa-answer"
                rows="3"
                placeholder="Enter the answer..."
                required
            >${answer}</textarea>
        </div>
        ${qaId ? `<input type="hidden" class="qa-id" value="${qaId}">` : ''}
    `;

    qaContainer.appendChild(qaCard);

    // Auto-resize textareas
    const textareas = qaCard.querySelectorAll('textarea');
    textareas.forEach(textarea => {
        textarea.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = this.scrollHeight + 'px';
        });
    });
}

window.removeQaPair = function(pairId) {
    const qaCard = document.querySelector(`[data-qa-id="${pairId}"]`);
    if (qaCard) {
        qaCard.remove();

        // Renumber remaining pairs
        const qaContainer = document.getElementById('qaContainer');
        if (qaContainer) {
            const pairs = qaContainer.querySelectorAll('.qa-pair-card');
            pairs.forEach((pair, index) => {
                const numberSpan = pair.querySelector('.qa-pair-number');
                if (numberSpan) {
                    numberSpan.textContent = `Q&A Pair #${index + 1}`;
                }
            });
            qaCount = pairs.length;
        }
    }
};

async function saveManualQA() {
    const qaContainer = document.getElementById('qaContainer');
    if (!qaContainer) return;

    const qaPairs = [];
    const qaCards = qaContainer.querySelectorAll('.qa-pair-card');

    if (qaCards.length === 0) {
        showAlert('Please add at least one Q&A pair', 'error');
        return;
    }

    let hasError = false;
    qaCards.forEach(card => {
        const questionInput = card.querySelector('.qa-question');
        const answerInput = card.querySelector('.qa-answer');
        const qaIdInput = card.querySelector('.qa-id');

        const question = questionInput ? questionInput.value.trim() : '';
        const answer = answerInput ? answerInput.value.trim() : '';
        const qaId = qaIdInput ? parseInt(qaIdInput.value) : null;

        if (!question || !answer) {
            hasError = true;
            if (!question) questionInput.classList.add('error');
            if (!answer) answerInput.classList.add('error');
        } else {
            questionInput.classList.remove('error');
            answerInput.classList.remove('error');

            const pair = { question, answer };
            if (qaId && !isNaN(qaId)) {
                pair.id = qaId;
            }
            qaPairs.push(pair);
        }
    });

    if (hasError) {
        showAlert('Please fill in all question and answer fields', 'error');
        return;
    }

    console.log('💾 Saving manual Q&A:', qaPairs);

    const saveManualBtn = document.getElementById('saveManualBtn');
    if (saveManualBtn) {
        saveManualBtn.disabled = true;
        saveManualBtn.innerHTML = '<span class="loading-spinner"></span><span>Saving & Training...</span>';
    }

    try {
        const response = await fetch(`/chatbot/train-manual/${CHATBOT_ID}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                qa_pairs: qaPairs,
                upload_mode: uploadMode
            })
        });

        const result = await response.json();

        if (result.success) {
            showAlert(result.message, 'success');
            setTimeout(() => {
                backToMainPage();
                loadQACards();
            }, 1200);
        } else {
            showAlert(result.message || 'Failed to save Q&A pairs', 'error');
            resetSaveManualButton();
        }
    } catch (error) {
        console.error('Manual save error:', error);
        showAlert('An error occurred while saving', 'error');
        resetSaveManualButton();
    }
}

function resetSaveManualButton() {
    const saveManualBtn = document.getElementById('saveManualBtn');
    if (saveManualBtn) {
        saveManualBtn.disabled = false;
        saveManualBtn.innerHTML = `
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <polyline points="20 6 9 17 4 12"></polyline>
            </svg>
            Save & Train Chatbot
        `;
    }
}

// ================================================================
// ✅ ADD-ON PAGINATION (NON-DESTRUCTIVE)
// ================================================================

/**
 * This pagination layer does NOT modify existing logic.
 * It only wraps renderQACards() safely.
 */

// ---- CONFIG ----
const QA_PAGE_SIZE = 10;
let qaCurrentPage = 1;

// ---- PRESERVE ORIGINAL FUNCTION ----
const _originalRenderQACards = window.renderQACards;

// ---- OVERRIDE SAFELY ----
window.renderQACards = function (qaPairs) {
    if (!Array.isArray(qaPairs)) return;

    // Save full filtered data
    window.__qaFullData = qaPairs;

    const totalPages = Math.ceil(qaPairs.length / QA_PAGE_SIZE);
    if (qaCurrentPage > totalPages) qaCurrentPage = 1;

    const start = (qaCurrentPage - 1) * QA_PAGE_SIZE;
    const end = start + QA_PAGE_SIZE;
    const pageData = qaPairs.slice(start, end);

    // Call original renderer
    _originalRenderQACards(pageData);

    // Render pagination UI
    renderQAPagination(totalPages);
};

// ---- PAGINATION UI ----
function renderQAPagination(totalPages) {
    let pagination = document.getElementById('qaPagination');

    if (!pagination) {
        pagination = document.createElement('div');
        pagination.id = 'qaPagination';
        pagination.className = 'qa-pagination';
        document.getElementById('qaCardsContainer')?.after(pagination);
    }

    if (totalPages <= 1) {
        pagination.innerHTML = '';
        return;
    }

    let html = `
        <button ${qaCurrentPage === 1 ? 'disabled' : ''}
            onclick="changeQAPage(${qaCurrentPage - 1})">Prev</button>
    `;

    for (let i = 1; i <= totalPages; i++) {
        html += `
            <button class="${i === qaCurrentPage ? 'active' : ''}"
                onclick="changeQAPage(${i})">${i}</button>
        `;
    }

    html += `
        <button ${qaCurrentPage === totalPages ? 'disabled' : ''}
            onclick="changeQAPage(${qaCurrentPage + 1})">Next</button>
    `;

    pagination.innerHTML = html;
}

// ---- PAGE CHANGE HANDLER ----
window.changeQAPage = function (page) {
    const data = window.__qaFullData || [];
    const totalPages = Math.ceil(data.length / QA_PAGE_SIZE);
    if (page < 1 || page > totalPages) return;

    qaCurrentPage = page;
    _originalRenderQACards(
        data.slice(
            (qaCurrentPage - 1) * QA_PAGE_SIZE,
            qaCurrentPage * QA_PAGE_SIZE
        )
    );
    renderQAPagination(totalPages);

    document.getElementById('qaCardsContainer')?.scrollIntoView({
        behavior: 'smooth',
        block: 'start'
    });
};


// ===== INITIALIZATION COMPLETE =====
console.log('Training page fully initialized');