// API Configuration
const API_BASE = 'http://localhost:8000/api';

// DOM Elements
const fileInput = document.getElementById('fileInput');
const uploadArea = document.getElementById('uploadArea');
const uploadStatus = document.getElementById('uploadStatus');
const wikiTopic = document.getElementById('wikiTopic');
const loadWikiBtn = document.getElementById('loadWikiBtn');

const chatMessages = document.getElementById('chatMessages');
const chatForm = document.getElementById('chatForm');
const questionInput = document.getElementById('questionInput');
const sendBtn = document.getElementById('sendBtn');
const docIndicator = document.getElementById('docIndicator');
const docName = document.getElementById('docName');

const historyList = document.getElementById('historyList');
const newChatBtn = document.getElementById('newChatBtn');
const loadingOverlay = document.getElementById('loadingOverlay');

// File Upload
uploadArea.addEventListener('click', () => fileInput.click());

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.style.borderColor = 'var(--primary)';
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.style.borderColor = 'var(--border)';
});

uploadArea.addEventListener('drop', async (e) => {
    e.preventDefault();
    uploadArea.style.borderColor = 'var(--border)';
    const file = e.dataTransfer.files[0];
    if (file) await uploadFile(file);
});

fileInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (file) await uploadFile(file);
});

async function uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        showLoading();
        uploadStatus.style.display = 'block';
        uploadStatus.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Uploading...';
        
        const response = await fetch(`${API_BASE}/documents/upload`, {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        hideLoading();
        
        if (response.ok) {
            uploadStatus.innerHTML = `<i class="fas fa-check-circle" style="color: var(--success)"></i> ${file.name} loaded (${data.chunks} chunks)`;
            enableChat(file.name);
            clearWelcome();
        } else {
            uploadStatus.innerHTML = `<i class="fas fa-exclamation-circle" style="color: var(--error)"></i> ${data.detail}`;
        }
    } catch (error) {
        hideLoading();
        uploadStatus.innerHTML = `<i class="fas fa-exclamation-circle" style="color: var(--error)"></i> Upload failed`;
    }
}

// Wikipedia Load
loadWikiBtn.addEventListener('click', async () => {
    const topic = wikiTopic.value.trim();
    if (!topic) return;
    
    try {
        showLoading();
        const response = await fetch(`${API_BASE}/documents/wikipedia?topic=${encodeURIComponent(topic)}`, {
            method: 'POST'
        });
        
        const data = await response.json();
        hideLoading();
        
        if (response.ok) {
            uploadStatus.style.display = 'block';
            uploadStatus.innerHTML = `<i class="fas fa-check-circle" style="color: var(--success)"></i> Wikipedia: ${topic} (${data.chunks} chunks)`;
            enableChat(`Wikipedia: ${topic}`);
            clearWelcome();
        } else {
            uploadStatus.innerHTML = `<i class="fas fa-exclamation-circle" style="color: var(--error)"></i> ${data.detail}`;
            uploadStatus.style.display = 'block';
        }
    } catch (error) {
        hideLoading();
        uploadStatus.innerHTML = `<i class="fas fa-exclamation-circle" style="color: var(--error)"></i> Load failed`;
        uploadStatus.style.display = 'block';
    }
});

// Chat
chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const question = questionInput.value.trim();
    if (!question) return;
    
    // Add user message
    addMessage('user', question);
    questionInput.value = '';
    
    try {
        const response = await fetch(`${API_BASE}/query`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            addMessage('assistant', data.answer, data.sources, data.confidence);
            loadHistory();
        } else {
            addMessage('assistant', `Error: ${data.detail}`, [], 0);
        }
    } catch (error) {
        addMessage('assistant', 'Network error. Please try again.', [], 0);
    }
});

// History
async function loadHistory() {
    try {
        const response = await fetch(`${API_BASE}/history`);
        
        if (response.ok) {
            const history = await response.json();
            historyList.innerHTML = '';
            
            history.forEach(item => {
                const div = document.createElement('div');
                div.className = 'history-item';
                div.innerHTML = `
                    <div class="question">${item.question}</div>
                    <div class="timestamp">${formatDate(item.timestamp)}</div>
                `;
                div.addEventListener('click', () => {
                    addMessage('user', item.question);
                    addMessage('assistant', item.answer, [], 0);
                });
                historyList.appendChild(div);
            });
        }
    } catch (error) {
        console.error('Failed to load history:', error);
    }
}

// UI Helpers
function showLoading() {
    loadingOverlay.style.display = 'flex';
}

function hideLoading() {
    loadingOverlay.style.display = 'none';
}

function enableChat(name) {
    questionInput.disabled = false;
    sendBtn.disabled = false;
    docIndicator.style.display = 'flex';
    docName.textContent = name;
}

function clearWelcome() {
    const welcome = chatMessages.querySelector('.welcome-message');
    if (welcome) welcome.remove();
}

function addMessage(role, text, sources = [], confidence = 0) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.innerHTML = role === 'user' ? '<i class="fas fa-user"></i>' : '<i class="fas fa-robot"></i>';
    
    const content = document.createElement('div');
    content.className = 'message-content';
    
    let html = `<div class="message-text">${text}</div>`;
    
    if (sources.length > 0) {
        html += `
            <div class="message-sources">
                <h4><i class="fas fa-book"></i> Sources</h4>
                ${sources.map((s, i) => `<span class="source-chip">Chunk ${i + 1}: ${s.substring(0, 50)}...</span>`).join('')}
            </div>
        `;
    }
    
    if (confidence > 0) {
        const level = confidence > 0.7 ? 'high' : confidence > 0.4 ? 'medium' : 'low';
        html += `
            <div class="message-meta">
                <span class="confidence-badge ${level}">
                    <i class="fas fa-chart-line"></i>
                    ${(confidence * 100).toFixed(0)}% confidence
                </span>
            </div>
        `;
    }
    
    content.innerHTML = html;
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(content);
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function formatDate(dateString) {
    const date = new Date(dateString);
    const now = new Date();
    const diff = now - date;
    const hours = Math.floor(diff / 3600000);
    
    if (hours < 1) return 'Just now';
    if (hours < 24) return `${hours}h ago`;
    return date.toLocaleDateString();
}

newChatBtn.addEventListener('click', () => {
    chatMessages.innerHTML = `
        <div class="welcome-message">
            <i class="fas fa-robot"></i>
            <h2>Welcome to RAG Chat</h2>
            <p>Upload a document or load a Wikipedia page to start asking questions</p>
            <div class="feature-cards">
                <div class="feature-card">
                    <i class="fas fa-file-pdf"></i>
                    <h4>Upload Documents</h4>
                    <p>PDF, TXT files</p>
                </div>
                <div class="feature-card">
                    <i class="fab fa-wikipedia-w"></i>
                    <h4>Wikipedia Search</h4>
                    <p>Any topic</p>
                </div>
                <div class="feature-card">
                    <i class="fas fa-comments"></i>
                    <h4>Smart Q&A</h4>
                    <p>Context-aware answers</p>
                </div>
            </div>
        </div>
    `;
    questionInput.disabled = true;
    sendBtn.disabled = true;
    docIndicator.style.display = 'none';
});
