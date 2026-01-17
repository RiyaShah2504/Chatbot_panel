// search-filter.js - Search & Filter

let allQAPairs = [];
let filteredQAPairs = [];
let currentFilter = 'all';
let currentSort = 'recent';
let searchQuery = '';

// ===== INITIALIZE SEARCH & FILTER =====
document.addEventListener('DOMContentLoaded', function() {
    initializeSearchFilter();
});

async function initializeSearchFilter() {
    const searchInput = document.getElementById('searchInput');
    const clearSearchBtn = document.getElementById('clearSearchBtn');
    const filterChips = document.querySelectorAll('.filter-chip');
    const sortSelect = document.getElementById('sortSelect');

    // Load all Q&A pairs
    await loadAllQAPairs();

    // Search input event
    if (searchInput) {
        searchInput.addEventListener('input', (e) => {
            searchQuery = e.target.value.toLowerCase();
            clearSearchBtn.style.display = searchQuery ? 'flex' : 'none';
            applyFiltersAndSearch();
        });
    }

    // Clear search button
    if (clearSearchBtn) {
        clearSearchBtn.addEventListener('click', () => {
            searchInput.value = '';
            searchQuery = '';
            clearSearchBtn.style.display = 'none';
            applyFiltersAndSearch();
        });
    }

    // Filter chips
    filterChips.forEach(chip => {
        chip.addEventListener('click', () => {
            filterChips.forEach(c => c.classList.remove('active'));
            chip.classList.add('active');
            currentFilter = chip.dataset.filter;
            applyFiltersAndSearch();
        });
    });

    // Sort select
    if (sortSelect) {
        sortSelect.addEventListener('change', (e) => {
            currentSort = e.target.value;
            applyFiltersAndSearch();
        });
    }
}

// ===== LOAD ALL Q&A PAIRS =====
async function loadAllQAPairs() {
    try {
        const response = await fetch(`/chatbot/get-qa-pairs/${CHATBOT_ID}`);
        const result = await response.json();

        if (result.success && result.qa_pairs) {
            allQAPairs = result.qa_pairs;
            applyFiltersAndSearch();
        }
    } catch (error) {
        console.error('Load Q&A error:', error);
        showAlert('Failed to load Q&A pairs', 'error');
    }
}

// ===== APPLY FILTERS & SEARCH =====
function applyFiltersAndSearch() {
    let results = [...allQAPairs];

    // Apply search filter
    if (searchQuery) {
        results = results.filter(qa => {
            return qa.question.toLowerCase().includes(searchQuery) ||
                   qa.answer.toLowerCase().includes(searchQuery);
        });
    }

    // Apply category filter
    if (currentFilter === 'recent') {
        results = results.filter(qa => {
            if (!qa.created_at) return false;
            const created = new Date(qa.created_at);
            const dayAgo = new Date(Date.now() - 24 * 60 * 60 * 1000);
            return created > dayAgo;
        });
    }

    // Apply sorting
    results = sortResults(results);

    // Update UI
    filteredQAPairs = results;
    renderSearchResults(results);
    updateResultsCount(results.length);
}

// ===== SORT RESULTS =====
function sortResults(results) {
    const sorted = [...results];

    switch (currentSort) {
        case 'oldest':
            sorted.sort((a, b) => new Date(a.created_at) - new Date(b.created_at));
            break;
        case 'az':
            sorted.sort((a, b) => a.question.localeCompare(b.question));
            break;
        case 'za':
            sorted.sort((a, b) => b.question.localeCompare(a.question));
            break;
        case 'recent':
        default:
            sorted.sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
    }

    return sorted;
}

// ===== RENDER SEARCH RESULTS =====
function renderSearchResults(results) {
    const container = document.getElementById('searchResultsContainer');
    const noResultsState = document.getElementById('noResultsState');

    if (!container) return;

    if (results.length === 0) {
        container.style.display = 'none';
        noResultsState.style.display = 'block';

        const message = document.getElementById('noResultsMessage');
        if (message) {
            if (searchQuery) {
                message.textContent = `No results found for "${searchQuery}"`;
            } else if (currentFilter === 'recent') {
                message.textContent = 'No recent Q&A pairs (last 24 hours)';
            } else {
                message.textContent = 'No Q&A pairs to display';
            }
        }
        return;
    }

    container.style.display = 'grid';
    noResultsState.style.display = 'none';

    container.innerHTML = results.map((qa) => {
        const isRecent = qa.created_at &&
            new Date(qa.created_at) > new Date(Date.now() - 24 * 60 * 60 * 1000);

        // Clean Q&A format
        let cleanQuestion = qa.question.replace(/^["']|["']$/g, '').replace(/^"tag":|^"patterns":|^"questions":|^"question":/i, '').trim();
        let cleanAnswer = qa.answer.replace(/^["']|["']$/g, '').replace(/^"responses":|^"response":|^"answers":|^"answer":/i, '').trim();

        return `
            <div class="search-result-card">
                <div class="result-badge ${isRecent ? 'recent' : ''}">
                    ${isRecent ? '🕐 Recent' : ''}
                </div>

                <div style="margin-bottom: 0.75rem;">
                    <div style="font-size: 0.75rem; font-weight: 600; color: #667eea; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 0.25rem;">Question</div>
                    <div class="result-question">${escapeHtml(cleanQuestion)}</div>
                </div>

                <div style="margin-bottom: 1rem;">
                    <div style="font-size: 0.75rem; font-weight: 600; color: #667eea; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 0.25rem;">Answer</div>
                    <div class="result-answer">
                        ${escapeHtml(cleanAnswer.substring(0, 120))}${cleanAnswer.length > 120 ? '...' : ''}
                    </div>
                </div>

                <div class="result-meta">
                    <div class="result-timestamp">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <circle cx="12" cy="12" r="10"></circle>
                            <polyline points="12 6 12 12 16 14"></polyline>
                        </svg>
                        <span>${formatTimestamp(qa.created_at)}</span>
                    </div>
                    <div class="result-actions">
                        <button class="result-action-btn" onclick="openEditModal(${qa.id}, '${qa.question.replace(/'/g, "\\'")}', '${qa.answer.replace(/'/g, "\\'")}')" title="Edit this Q&A pair">
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"></path>
                                <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"></path>
                            </svg>
                            Edit
                        </button>
                        <button class="result-action-btn danger" onclick="confirmDeleteQAPair(${qa.id})" title="Delete this Q&A pair">
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <polyline points="3 6 5 6 21 6"></polyline>
                                <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                            </svg>
                            Del
                        </button>
                    </div>
                </div>
            </div>
        `;
    }).join('');
}

// ===== OPEN EDIT MODAL =====
function openEditModal(qaId, question, answer) {
    // Remove existing modal if any
    const existingModal = document.getElementById('editQAModal');
    if (existingModal) existingModal.remove();

    const modal = document.createElement('div');
    modal.id = 'editQAModal';
    modal.className = 'modal';
    modal.innerHTML = `
        <div class="modal-overlay"></div>
        <div class="modal-content">
            <div class="modal-header">
                <h2>✏️ Edit Q&A Pair</h2>
                <button class="modal-close" type="button" onclick="closeEditModal()">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <line x1="18" y1="6" x2="6" y2="18"></line>
                        <line x1="6" y1="6" x2="18" y2="18"></line>
                    </svg>
                </button>
            </div>
            <div class="modal-body">
                <form id="editQAForm" onsubmit="saveEditQAPair(event, ${qaId})">
                    <div class="form-group">
                        <label for="editQuestion">Question <span style="color: #dc2626;">*</span></label>
                        <textarea
                            id="editQuestion"
                            class="form-input"
                            maxlength="500"
                            placeholder="Enter question"
                            required>${unescapeHtml(question)}</textarea>
                        <small style="color: #6b7280; font-size: 0.75rem;">
                            <span id="qCount">${question.length}</span>/500 characters
                        </small>
                    </div>

                    <div class="form-group">
                        <label for="editAnswer">Answer <span style="color: #dc2626;">*</span></label>
                        <textarea
                            id="editAnswer"
                            class="form-input"
                            maxlength="1000"
                            placeholder="Enter answer"
                            required>${unescapeHtml(answer)}</textarea>
                        <small style="color: #6b7280; font-size: 0.75rem;">
                            <span id="aCount">${answer.length}</span>/1000 characters
                        </small>
                    </div>

                    <div style="display: flex; gap: 1rem; margin-top: 1.5rem;">
                        <button type="submit" class="btn-primary" style="flex: 1;">
                            ✓ Save Changes
                        </button>
                        <button type="button" class="btn-secondary" style="flex: 1;" onclick="closeEditModal()">
                            Cancel
                        </button>
                    </div>
                </form>
            </div>
        </div>
    `;

    document.body.appendChild(modal);
    modal.style.display = 'flex';

    // Setup character counters
    const editQuestion = document.getElementById('editQuestion');
    const editAnswer = document.getElementById('editAnswer');

    editQuestion.addEventListener('input', (e) => {
        document.getElementById('qCount').textContent = e.target.value.length;
    });

    editAnswer.addEventListener('input', (e) => {
        document.getElementById('aCount').textContent = e.target.value.length;
    });

    // Close on overlay click
    const overlay = modal.querySelector('.modal-overlay');
    overlay.addEventListener('click', (e) => {
        if (e.target === overlay) {
            closeEditModal();
        }
    });
}

// ===== CLOSE EDIT MODAL =====
function closeEditModal() {
    const modal = document.getElementById('editQAModal');
    if (modal) {
        modal.style.display = 'none';
        setTimeout(() => {
            if (modal && modal.parentNode) {
                modal.remove();
            }
        }, 200);
    }
}

// ===== SAVE EDITED Q&A PAIR =====
async function saveEditQAPair(event, qaId) {
    event.preventDefault();

    const question = document.getElementById('editQuestion').value.trim();
    const answer = document.getElementById('editAnswer').value.trim();

    if (!question || !answer) {
        showAlert('Please fill in both fields', 'error');
        return;
    }

    if (question.length > 500 || answer.length > 1000) {
        showAlert('Text exceeds maximum length', 'error');
        return;
    }

    try {
        const response = await fetch(`/chatbot/update-qa-pair/${qaId}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question, answer })
        });

        const result = await response.json();

        if (result.success) {
            showAlert('✅ Q&A pair updated successfully!', 'success');
            closeEditModal();
            setTimeout(() => loadAllQAPairs(), 600);
        } else {
            showAlert(result.message || 'Failed to update Q&A pair', 'error');
        }
    } catch (error) {
        console.error('Update error:', error);
        showAlert('Error updating Q&A pair', 'error');
    }
}

// ===== CONFIRM DELETE Q&A PAIR =====
function confirmDeleteQAPair(qaId) {
    if (confirm('Are you sure you want to delete this Q&A pair? This action cannot be undone.')) {
        deleteQAPairFromList(qaId);
    }
}

// ===== DELETE Q&A PAIR FROM LIST =====
async function deleteQAPairFromList(qaId) {
    try {
        const response = await fetch(`/chatbot/delete-qa-pair/${qaId}`, { method: 'POST' });
        const result = await response.json();

        if (result.success) {
            showAlert('✅ Q&A pair deleted successfully', 'success');
            await loadAllQAPairs();
        } else {
            showAlert(result.message || 'Failed to delete Q&A pair', 'error');
        }
    } catch (error) {
        console.error('Delete error:', error);
        showAlert('An error occurred while deleting', 'error');
    }
}

// ===== UPDATE RESULTS COUNT =====
function updateResultsCount(count) {
    const resultsCount = document.getElementById('resultsCount');
    const totalCount = document.getElementById('totalCount');

    if (resultsCount) resultsCount.textContent = count;
    if (totalCount) totalCount.textContent = allQAPairs.length;
}

// ===== UTILITY FUNCTIONS =====
function formatTimestamp(timestamp) {
    if (!timestamp) return 'Recently';

    const date = new Date(timestamp);
    if (isNaN(date.getTime())) return 'Recently';

    const now = new Date();
    const diffMs = now - date;
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays < 7) return `${diffDays}d ago`;

    return date.toLocaleDateString();
}

function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function unescapeHtml(text) {
    if (!text) return '';
    const textarea = document.createElement('textarea');
    textarea.innerHTML = text;
    return textarea.value;
}