/**
 * ChestAI — Frontend Application Logic
 *
 * Handles file upload (drag-and-drop + browse), image preview,
 * API communication with FastAPI backend, and result rendering.
 *
 * Backend endpoint: POST http://127.0.0.1:8000/predict
 * Request: multipart/form-data with field "file"
 * Response: { filename, prediction, confidence }
 */

(() => {
  'use strict';

  // ── Configuration ──────────────────────────────────────────
  const API_URL = 'http://127.0.0.1:8000/predict';
  const MAX_FILE_SIZE_MB = 20;
  const ACCEPTED_TYPES = ['image/png', 'image/jpeg', 'image/jpg', 'image/webp'];

  // ── DOM References ─────────────────────────────────────────
  const dom = {
    // Sections
    uploadArea:    document.getElementById('upload-area'),
    previewArea:   document.getElementById('preview-area'),
    loadingArea:   document.getElementById('loading-area'),
    resultArea:    document.getElementById('result-area'),
    errorArea:     document.getElementById('error-area'),

    // Upload
    dropzone:      document.getElementById('upload-dropzone'),
    fileInput:     document.getElementById('file-input'),
    browseBtn:     document.getElementById('browse-btn'),

    // Preview
    previewImage:  document.getElementById('preview-image'),
    fileName:      document.getElementById('file-name'),
    fileSize:      document.getElementById('file-size'),
    removeBtn:     document.getElementById('remove-btn'),
    analyzeBtn:    document.getElementById('analyze-btn'),

    // Loading
    loadingBarFill: document.getElementById('loading-bar-fill'),

    // Result
    resultPredCard:  document.getElementById('result-prediction-card'),
    resultPrediction: document.getElementById('result-prediction'),
    confidenceValue: document.getElementById('confidence-value'),
    confidenceBar:   document.getElementById('confidence-bar-fill'),
    resultFilename:  document.getElementById('result-filename'),
    resultTimestamp: document.getElementById('result-timestamp'),
    resultIconWrap:  document.getElementById('result-icon-wrapper'),
    newAnalysisBtn:  document.getElementById('new-analysis-btn'),

    // Error
    errorMessage:  document.getElementById('error-message'),
    retryBtn:      document.getElementById('retry-btn'),

    // Navbar
    navbar:        document.getElementById('navbar'),
  };

  // ── State ──────────────────────────────────────────────────
  let selectedFile = null;

  // ── Helpers ────────────────────────────────────────────────

  /**
   * Show one section, hide the rest inside the upload card.
   */
  function showSection(sectionEl) {
    [dom.uploadArea, dom.previewArea, dom.loadingArea, dom.resultArea, dom.errorArea].forEach(el => {
      el.classList.add('hidden');
    });
    sectionEl.classList.remove('hidden');
  }

  /**
   * Formats file size in human-readable form.
   */
  function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
  }

  /**
   * Validates an uploaded file.
   * Returns null if valid, or an error message string.
   */
  function validateFile(file) {
    if (!file) return 'No file selected.';
    if (!ACCEPTED_TYPES.includes(file.type)) {
      return `Unsupported format "${file.type.split('/')[1] || 'unknown'}". Please upload PNG, JPG, or WEBP.`;
    }
    if (file.size > MAX_FILE_SIZE_MB * 1024 * 1024) {
      return `File is too large (${formatFileSize(file.size)}). Maximum size is ${MAX_FILE_SIZE_MB} MB.`;
    }
    return null;
  }

  /**
   * Returns a condition key (normal / pneumonia / tuberculosis) from prediction text.
   */
  function conditionKey(prediction) {
    const lower = prediction.toLowerCase();
    if (lower.includes('normal')) return 'normal';
    if (lower.includes('pneumonia')) return 'pneumonia';
    if (lower.includes('tuberculosis') || lower.includes('tb')) return 'tuberculosis';
    return 'unknown';
  }

  /**
   * Returns confidence level (high/medium/low) from a percentage number.
   */
  function confidenceLevel(pct) {
    if (pct >= 70) return 'high';
    if (pct >= 40) return 'medium';
    return 'low';
  }

  /**
   * Parses confidence string ("74.62%") into a number (74.62).
   */
  function parseConfidence(str) {
    return parseFloat(str.replace('%', '')) || 0;
  }

  /**
   * Formats current time as a readable string.
   */
  function formatTimestamp() {
    const now = new Date();
    return now.toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    });
  }

  // ── File Selection ─────────────────────────────────────────

  function handleFileSelected(file) {
    const error = validateFile(file);
    if (error) {
      showError(error);
      return;
    }

    selectedFile = file;

    // Set file info
    dom.fileName.textContent = file.name;
    dom.fileSize.textContent = formatFileSize(file.size);

    // Generate preview
    const reader = new FileReader();
    reader.onload = (e) => {
      dom.previewImage.src = e.target.result;
      showSection(dom.previewArea);
    };
    reader.onerror = () => {
      showError('Failed to read the selected file. Please try again.');
    };
    reader.readAsDataURL(file);
  }

  // ── Drag & Drop ────────────────────────────────────────────

  function initDragDrop() {
    const dz = dom.dropzone;
    let dragCounter = 0;

    dz.addEventListener('dragenter', (e) => {
      e.preventDefault();
      dragCounter++;
      dz.classList.add('drag-over');
    });

    dz.addEventListener('dragleave', (e) => {
      e.preventDefault();
      dragCounter--;
      if (dragCounter <= 0) {
        dragCounter = 0;
        dz.classList.remove('drag-over');
      }
    });

    dz.addEventListener('dragover', (e) => {
      e.preventDefault();
    });

    dz.addEventListener('drop', (e) => {
      e.preventDefault();
      dragCounter = 0;
      dz.classList.remove('drag-over');
      const file = e.dataTransfer.files[0];
      if (file) handleFileSelected(file);
    });

    // Click to browse
    dz.addEventListener('click', () => dom.fileInput.click());

    // Keyboard accessibility
    dz.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        dom.fileInput.click();
      }
    });
  }

  // ── API Communication ──────────────────────────────────────

  async function analyzeImage() {
    if (!selectedFile) return;

    showSection(dom.loadingArea);

    // Reset loading bar animation
    dom.loadingBarFill.style.animation = 'none';
    // Force reflow to restart animation
    void dom.loadingBarFill.offsetHeight;
    dom.loadingBarFill.style.animation = '';

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch(API_URL, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        let errorMsg = `Server returned status ${response.status}.`;
        try {
          const errData = await response.json();
          if (errData.detail) errorMsg = errData.detail;
          else if (errData.message) errorMsg = errData.message;
        } catch {
          // Use default error message
        }
        throw new Error(errorMsg);
      }

      const data = await response.json();
      displayResult(data);

    } catch (err) {
      if (err.name === 'TypeError' && err.message.includes('fetch')) {
        showError('Could not connect to the backend server. Please ensure the FastAPI server is running at ' + API_URL);
      } else {
        showError(err.message || 'An unexpected error occurred during analysis.');
      }
    }
  }

  // ── Result Rendering ───────────────────────────────────────

  function displayResult(data) {
    const { filename, prediction, confidence } = data;
    const key = conditionKey(prediction);
    const confNum = parseConfidence(confidence);
    const level = confidenceLevel(confNum);

    // Prediction card
    dom.resultPredCard.setAttribute('data-condition', key);
    dom.resultPrediction.textContent = prediction;

    // Confidence
    dom.confidenceValue.textContent = confidence;
    dom.confidenceBar.style.width = '0%';
    dom.confidenceBar.setAttribute('data-level', level);

    // Animate confidence bar after a short delay
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        dom.confidenceBar.style.width = confNum + '%';
      });
    });

    // Meta
    dom.resultFilename.textContent = filename || selectedFile.name;
    dom.resultTimestamp.textContent = formatTimestamp();

    // Result icon color by condition
    if (key === 'normal') {
      dom.resultIconWrap.style.background = 'var(--clr-success-soft)';
      dom.resultIconWrap.style.color = 'var(--clr-success)';
    } else if (key === 'pneumonia') {
      dom.resultIconWrap.style.background = 'var(--clr-warning-soft)';
      dom.resultIconWrap.style.color = 'var(--clr-warning)';
    } else if (key === 'tuberculosis') {
      dom.resultIconWrap.style.background = 'var(--clr-danger-soft)';
      dom.resultIconWrap.style.color = 'var(--clr-danger)';
    }

    showSection(dom.resultArea);
  }

  // ── Error Rendering ────────────────────────────────────────

  function showError(message) {
    dom.errorMessage.textContent = message;
    showSection(dom.errorArea);
  }

  // ── Reset to Upload ────────────────────────────────────────

  function resetToUpload() {
    selectedFile = null;
    dom.fileInput.value = '';
    dom.previewImage.src = '';
    dom.confidenceBar.style.width = '0%';
    showSection(dom.uploadArea);
  }

  // ── Navbar Scroll Effect ───────────────────────────────────

  function initNavbarScroll() {
    let lastScroll = 0;
    window.addEventListener('scroll', () => {
      const current = window.scrollY;
      if (current > 80) {
        dom.navbar.style.background = 'rgba(10, 14, 26, 0.92)';
      } else {
        dom.navbar.style.background = 'rgba(10, 14, 26, 0.7)';
      }
      lastScroll = current;
    }, { passive: true });
  }

  // ── Event Listeners ────────────────────────────────────────

  function initEvents() {
    // Browse button
    dom.browseBtn.addEventListener('click', () => dom.fileInput.click());

    // File input change
    dom.fileInput.addEventListener('change', (e) => {
      const file = e.target.files[0];
      if (file) handleFileSelected(file);
    });

    // Remove image
    dom.removeBtn.addEventListener('click', resetToUpload);

    // Analyze button
    dom.analyzeBtn.addEventListener('click', analyzeImage);

    // New analysis
    dom.newAnalysisBtn.addEventListener('click', resetToUpload);

    // Retry
    dom.retryBtn.addEventListener('click', () => {
      if (selectedFile) {
        analyzeImage();
      } else {
        resetToUpload();
      }
    });
  }

  // ── Initialize ─────────────────────────────────────────────

  function init() {
    initDragDrop();
    initEvents();
    initNavbarScroll();
  }

  // Start when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
