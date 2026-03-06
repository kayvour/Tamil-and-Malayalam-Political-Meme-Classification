// ============================================================================
// STATE MANAGEMENT
// ============================================================================

let selectedFile = null;
let isProcessing = false;
let selectedModel = 'zero_shot_clip';
let availableModels = {};

// ============================================================================
// DOM ELEMENTS
// ============================================================================

const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const browseBtn = document.getElementById('browseBtn');
const previewImage = document.getElementById('previewImage');
const contentGrid = document.getElementById('contentGrid');
const classifyBtn = document.getElementById('classifyBtn');
const clearBtn = document.getElementById('clearBtn');
const loadingCard = document.getElementById('loadingCard');
const resultsContainer = document.getElementById('resultsContainer');
const statusBadge = document.getElementById('statusBadge');

// Result elements
const level1Label = document.getElementById('level1Label');
const level1Confidence = document.getElementById('level1Confidence');
const level1Bar = document.getElementById('level1Bar');
const level2Label = document.getElementById('level2Label');
const level2Confidence = document.getElementById('level2Confidence');
const level2Bar = document.getElementById('level2Bar');
const ocrText = document.getElementById('ocrText');
const modelOptions = document.getElementById('modelOptions');
const modelUsedName = document.getElementById('modelUsedName');

// ============================================================================
// INITIALIZATION
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
    checkServerStatus();
    loadModels();
    initializeEventListeners();
});

// ============================================================================
// SERVER STATUS
// ============================================================================

async function checkServerStatus() {
    try {
        const response = await fetch('/api/status');
        const data = await response.json();
        
        if (data.status === 'online') {
            updateStatusBadge('Ready', true);
        } else {
            updateStatusBadge('Offline', false);
        }
    } catch (error) {
        updateStatusBadge('Error', false);
        console.error('Server status check failed:', error);
    }
}

function updateStatusBadge(text, online) {
    const statusDot = statusBadge.querySelector('.status-dot');
    const statusText = statusBadge.querySelector('.status-text');
    
    statusText.textContent = text;
    statusDot.style.background = online ? 'var(--success)' : 'var(--danger)';
}

// ============================================================================
// MODEL LOADING
// ============================================================================

async function loadModels() {
    try {
        const response = await fetch('/api/models');
        const data = await response.json();
        availableModels = data.models;
        selectedModel = data.default;
        renderModelOptions();
    } catch (error) {
        console.error('Failed to load models:', error);
        modelOptions.innerHTML = '<p class="model-error">Could not load models.</p>';
    }
}

function renderModelOptions() {
    modelOptions.innerHTML = '';
    for (const [key, info] of Object.entries(availableModels)) {
        const isReady = info.status === 'ready';
        const isSelected = key === selectedModel;

        const option = document.createElement('label');
        option.className = `model-option${isSelected ? ' selected' : ''}${!isReady ? ' disabled' : ''}`;
        option.innerHTML = `
            <input type="radio" name="model" value="${key}" ${isSelected ? 'checked' : ''} ${!isReady ? 'disabled' : ''}>
            <div class="model-option-content">
                <div class="model-option-header">
                    <span class="model-option-name">${info.name}</span>
                    <span class="badge ${isReady ? 'badge-green' : 'badge-orange'}">${isReady ? 'Ready' : 'Not Trained'}</span>
                </div>
                <p class="model-option-desc">${info.description}</p>
            </div>
        `;
        if (isReady) {
            option.addEventListener('click', () => selectModel(key));
        }
        modelOptions.appendChild(option);
    }
}

function selectModel(key) {
    selectedModel = key;
    // Update visual selection
    document.querySelectorAll('.model-option').forEach(opt => opt.classList.remove('selected'));
    const radio = document.querySelector(`input[name="model"][value="${key}"]`);
    if (radio) {
        radio.checked = true;
        radio.closest('.model-option').classList.add('selected');
    }
}

// ============================================================================
// EVENT LISTENERS
// ============================================================================

function initializeEventListeners() {
    // Browse button
    browseBtn.addEventListener('click', () => fileInput.click());
    
    // Upload area click
    uploadArea.addEventListener('click', () => fileInput.click());
    
    // File input change
    fileInput.addEventListener('change', handleFileSelect);
    
    // Drag and drop
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);
    
    // Classify button
    classifyBtn.addEventListener('click', classifyImage);
    
    // Clear button
    clearBtn.addEventListener('click', clearAll);
}

// ============================================================================
// FILE HANDLING
// ============================================================================

function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file && isValidImageFile(file)) {
        loadImage(file);
    } else {
        showError('Please select a valid image file (JPG, PNG, GIF, BMP)');
    }
}

function handleDragOver(event) {
    event.preventDefault();
    uploadArea.classList.add('dragover');
}

function handleDragLeave(event) {
    event.preventDefault();
    uploadArea.classList.remove('dragover');
}

function handleDrop(event) {
    event.preventDefault();
    uploadArea.classList.remove('dragover');
    
    const file = event.dataTransfer.files[0];
    if (file && isValidImageFile(file)) {
        loadImage(file);
    } else {
        showError('Please drop a valid image file (JPG, PNG, GIF, BMP)');
    }
}

function isValidImageFile(file) {
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/bmp'];
    return validTypes.includes(file.type);
}

function loadImage(file) {
    selectedFile = file;
    
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        contentGrid.style.display = 'grid';
        resultsContainer.style.display = 'none';
        loadingCard.style.display = 'none';
    };
    reader.readAsDataURL(file);
}

// ============================================================================
// CLASSIFICATION
// ============================================================================

async function classifyImage() {
    if (!selectedFile || isProcessing) return;
    
    isProcessing = true;
    classifyBtn.disabled = true;
    loadingCard.style.display = 'block';
    resultsContainer.style.display = 'none';
    updateStatusBadge('Processing...', true);
    
    try {
        const formData = new FormData();
        formData.append('image', selectedFile);
        formData.append('model', selectedModel);
        
        const response = await fetch('/api/classify', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            displayResults(data);
        } else {
            showError(data.error || 'Classification failed');
        }
    } catch (error) {
        showError('Network error. Please check if the server is running.');
        console.error('Classification error:', error);
    } finally {
        isProcessing = false;
        classifyBtn.disabled = false;
        loadingCard.style.display = 'none';
        updateStatusBadge('Ready', true);
    }
}

// ============================================================================
// DISPLAY RESULTS
// ============================================================================

function displayResults(data) {
    // Level 1
    level1Label.textContent = data.level1.label;
    level1Confidence.textContent = `${data.level1.confidence}%`;
    level1Bar.style.width = `${data.level1.confidence}%`;
    
    // Level 2
    level2Label.textContent = data.level2.label;
    level2Confidence.textContent = `${data.level2.confidence}%`;
    level2Bar.style.width = `${data.level2.confidence}%`;
    
    // OCR Text
    ocrText.textContent = data.ocr_text || 'No text detected';
    
    // Model used
    modelUsedName.textContent = data.model_used || selectedModel;
    
    // Show results
    resultsContainer.style.display = 'block';
    
    // Update status
    updateStatusBadge(`Classified: ${data.level1.label}`, true);
}

// ============================================================================
// CLEAR & RESET
// ============================================================================

function clearAll() {
    selectedFile = null;
    fileInput.value = '';
    previewImage.src = '';
    contentGrid.style.display = 'none';
    resultsContainer.style.display = 'none';
    loadingCard.style.display = 'none';
    
    // Reset results
    level1Label.textContent = '—';
    level1Confidence.textContent = '—';
    level1Bar.style.width = '0%';
    level2Label.textContent = '—';
    level2Confidence.textContent = '—';
    level2Bar.style.width = '0%';
    ocrText.textContent = '—';
    
    updateStatusBadge('Ready', true);
}

// ============================================================================
// ERROR HANDLING
// ============================================================================

function showError(message) {
    alert(`Error: ${message}`);
}
