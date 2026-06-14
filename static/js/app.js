/* ==========================================================================
   DERMVISION AI FRONTEND CONTROLLER
   ========================================================================== */

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const browseBtn = document.getElementById('browse-btn');
    const uploadPrompt = document.getElementById('upload-prompt');
    const uploadPreview = document.getElementById('upload-preview');
    const previewImg = document.getElementById('preview-img');
    const loadingOverlay = document.getElementById('loading-overlay');
    const loadingStatus = document.getElementById('loading-status');
    
    // Panel Elements
    const preprocessingPanel = document.getElementById('preprocessing-panel');
    const preprocessingTabs = document.getElementById('preprocessing-tabs');
    const tabDisplayImg = document.getElementById('tab-display-img');
    const stepDesc = document.getElementById('step-desc');
    const welcomeView = document.getElementById('welcome-view');
    const resultsView = document.getElementById('results-view');
    const demoAlert = document.getElementById('demo-alert');
    
    // Diagnostics Elements
    const resultTitle = document.getElementById('result-title');
    const resultType = document.getElementById('result-type');
    const riskBadge = document.getElementById('risk-badge');
    const resultConfidence = document.getElementById('result-confidence');
    const confidenceRing = document.getElementById('confidence-ring');
    const resultDesc = document.getElementById('result-desc');
    const distributionList = document.getElementById('distribution-list');
    
    // Decision Support Elements
    const decisionTabs = document.querySelector('.decision-tabs');
    const listSkincare = document.getElementById('list-skincare');
    const listPrecautions = document.getElementById('list-precautions');
    const textMedical = document.getElementById('text-medical');
    const warningBox = document.getElementById('warning-box');
    
    // Control Buttons
    const resetBtn = document.getElementById('reset-btn');

    // History Elements
    const historyDrawer = document.getElementById('history-drawer');
    const historyToggleBtn = document.getElementById('history-toggle-btn');
    const closeDrawerBtn = document.getElementById('close-drawer-btn');
    const drawerOverlay = document.getElementById('drawer-overlay');
    const clearHistoryBtn = document.getElementById('clear-history-btn');
    const historyListContainer = document.getElementById('history-list-container');
    const emptyHistoryMsg = document.getElementById('empty-history-msg');

    // Global variable to hold response images
    let predictionImages = {};
    
    // Descriptions for each preprocessing stage
    const stepDescriptions = {
        original: "Original dermoscopic skin lesion image uploaded for screening.",
        grayscale: "Grayscale conversion simplifies the color space to prepare the image for intensity-based morphological filters.",
        blackhat: "Blackhat morphological transformation highlights narrow dark features (like hairs) that are darker than their surroundings.",
        hair_mask: "Binary thresholding creates a precise pixel-map segmenting hair structures from the underlying skin lesion.",
        hair_removed: "Telea inpainting reconstructs hair-covered pixels using the mathematical average of surrounding skin texture, eliminating occlusion.",
        enhanced: "Contrast Limited Adaptive Histogram Equalization is applied to the L channel in LAB color space to reveal boundary and color variation without introducing noise."
    };

    // Circular Progress Ring Math
    const radius = confidenceRing.r.baseVal.value;
    const circumference = radius * 2 * Math.PI;
    confidenceRing.style.strokeDasharray = `${circumference} ${circumference}`;
    setProgress(0);

    function setProgress(percent) {
        const offset = circumference - (percent / 100) * circumference;
        confidenceRing.style.strokeDashoffset = offset;
    }

    // ==========================================================================
    // UPLOAD & EVENT LISTENERS
    // ==========================================================================

    // History Drawer listeners
    historyToggleBtn.addEventListener('click', () => {
        historyDrawer.classList.add('open');
        loadHistory();
    });

    closeDrawerBtn.addEventListener('click', () => {
        historyDrawer.classList.remove('open');
    });

    drawerOverlay.addEventListener('click', () => {
        historyDrawer.classList.remove('open');
    });

    clearHistoryBtn.addEventListener('click', () => {
        if (confirm('Are you sure you want to clear all scan history?')) {
            clearHistory();
        }
    });

    // Browse button triggers file input
    browseBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.click();
    });

    dropZone.addEventListener('click', () => {
        if (uploadPreview.classList.contains('hidden')) {
            fileInput.click();
        }
    });

    // Drag and drop events
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove('dragover');
        }, false);
    });

    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    // Handle selected file
    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please select an image file (PNG/JPG).');
            return;
        }

        // Show preview
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImg.src = e.target.result;
            uploadPrompt.classList.add('hidden');
            uploadPreview.classList.remove('hidden');
            loadingOverlay.classList.remove('hidden');
            
            // Start pipeline simulation text, then upload
            animateLoadingStatus(file);
        };
        reader.readAsDataURL(file);
    }

    // Loading status animation
    function animateLoadingStatus(file) {
        const statuses = [
            "Morphological Dull Razor: Gray conversion...",
            "Morphological Dull Razor: Hair detection...",
            "Morphological Dull Razor: Telea inpainting...",
            "Lesion enhancement: CIELAB CLAHE scaling...",
            "Transfer Learning: Extracting feature maps...",
            "Attention Model: Squeeze-and-Excitation reweighting..."
        ];

        let index = 0;
        loadingStatus.textContent = statuses[0];
        
        const interval = setInterval(() => {
            index++;
            if (index < statuses.length) {
                loadingStatus.textContent = statuses[index];
            } else {
                clearInterval(interval);
            }
        }, 800);

        // Actually upload image to backend
        uploadImage(file, interval);
    }

    // Upload image via Ajax
    function uploadImage(file, interval) {
        const formData = new FormData();
        formData.append('image', file);

        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Server error: ' + response.statusText);
            }
            return response.json();
        })
        .then(data => {
            clearInterval(interval);
            if (data.success) {
                renderResults(data);
                loadHistory(); // Refresh history
            } else {
                alert('Analysis failed: ' + data.error);
                resetApp();
            }
        })
        .catch(error => {
            clearInterval(interval);
            console.error('Error:', error);
            alert('An error occurred during analysis: ' + error.message);
            resetApp();
        });
    }

    // ==========================================================================
    // RESULT RENDERING
    // ==========================================================================

    function renderResults(data) {
        loadingOverlay.classList.add('hidden');
        welcomeView.classList.add('hidden');
        resultsView.classList.remove('hidden');
        preprocessingPanel.classList.remove('hidden');
        
        // Save images for tab switcher
        predictionImages = data.images;
        
        // Load original step by default in preprocessing visualizer
        setPreprocessingTab('original');

        // Set Diagnosis details
        resultTitle.textContent = data.predicted_class;
        resultType.textContent = data.type;
        resultDesc.textContent = data.description;
        
        // Confidence ring & score
        resultConfidence.textContent = Math.round(data.probability);
        setProgress(data.probability);

        // Risk level badge
        riskBadge.className = 'diagnostic-badge'; // reset
        const risk = data.risk_level.toLowerCase();
        if (risk.includes('low')) {
            riskBadge.classList.add('risk-low');
            riskBadge.textContent = 'Low Risk';
            warningBox.className = 'warning-alert-box';
        } else if (risk.includes('medium') && !risk.includes('high')) {
            riskBadge.classList.add('risk-medium');
            riskBadge.textContent = 'Medium Risk';
            warningBox.className = 'warning-alert-box warning-box-medium';
        } else {
            riskBadge.classList.add('risk-high');
            riskBadge.textContent = 'High Risk';
            warningBox.className = 'warning-alert-box warning-box-high';
        }

        // Demo Mode Banner
        if (data.demo_mode) {
            demoAlert.classList.remove('hidden');
        } else {
            demoAlert.classList.add('hidden');
        }

        // Render Probability distribution chart list
        distributionList.innerHTML = '';
        data.distribution.forEach(item => {
            const container = document.createElement('div');
            container.className = 'distribution-item';
            
            const label = document.createElement('div');
            label.className = 'dist-label';
            
            const nameSpan = document.createElement('span');
            nameSpan.className = 'dist-name';
            nameSpan.textContent = item.name;
            
            const valueSpan = document.createElement('span');
            valueSpan.className = 'dist-value';
            valueSpan.textContent = `${item.probability}%`;
            
            label.appendChild(nameSpan);
            label.appendChild(valueSpan);
            
            const bgBar = document.createElement('div');
            bgBar.className = 'dist-bar-bg';
            
            const fillBar = document.createElement('div');
            fillBar.className = 'dist-bar-fill';
            // Trigger animation on next tick
            setTimeout(() => {
                fillBar.style.width = `${item.probability}%`;
            }, 50);

            bgBar.appendChild(fillBar);
            container.appendChild(label);
            container.appendChild(bgBar);
            distributionList.appendChild(container);
        });

        // Render Decision Support Lists
        listSkincare.innerHTML = '';
        data.skincare.forEach(item => {
            const li = document.createElement('li');
            li.textContent = item;
            listSkincare.appendChild(li);
        });

        listPrecautions.innerHTML = '';
        data.precautions.forEach(item => {
            const li = document.createElement('li');
            li.textContent = item;
            listPrecautions.appendChild(li);
        });

        textMedical.textContent = data.medical_advice;
    }

    // ==========================================================================
    // TAB INTERACTIONS
    // ==========================================================================

    // Preprocessing tabs click event
    preprocessingTabs.addEventListener('click', (e) => {
        const button = e.target.closest('.tab-btn');
        if (!button) return;
        
        // Remove active class from all buttons
        preprocessingTabs.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
        
        // Add active class to clicked button
        button.classList.add('active');
        
        // Set tab step
        const step = button.getAttribute('data-step');
        setPreprocessingTab(step);
    });

    function setPreprocessingTab(step) {
        if (!predictionImages[step]) return;
        
        // Set image source
        tabDisplayImg.src = `data:image/png;base64,${predictionImages[step]}`;
        
        // Set explanation
        stepDesc.textContent = stepDescriptions[step];
    }

    // Decision support tabs click event
    decisionTabs.addEventListener('click', (e) => {
        const button = e.target.closest('.decision-tab-btn');
        if (!button) return;

        // Remove active class
        decisionTabs.querySelectorAll('.decision-tab-btn').forEach(btn => btn.classList.remove('active'));
        document.querySelectorAll('.decision-panel').forEach(panel => panel.classList.remove('active'));

        // Add active class
        button.classList.add('active');
        const section = button.getAttribute('data-section');
        document.getElementById(`panel-${section}`).classList.add('active');
    });

    // ==========================================================================
    // RESET & RE-ANALYZE
    // ==========================================================================

    resetBtn.addEventListener('click', resetApp);

    function resetApp() {
        // Clear inputs
        fileInput.value = '';
        
        // Reset previews
        previewImg.src = '';
        uploadPreview.classList.add('hidden');
        uploadPrompt.classList.remove('hidden');
        loadingOverlay.classList.add('hidden');
        
        // Reset views
        welcomeView.classList.remove('hidden');
        resultsView.classList.add('hidden');
        preprocessingPanel.classList.add('hidden');
        
        // Clear global prediction data
        predictionImages = {};
        
        // Reset active tabs
        preprocessingTabs.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
        preprocessingTabs.querySelector('[data-step="original"]').classList.add('active');
        
        decisionTabs.querySelectorAll('.decision-tab-btn').forEach(btn => btn.classList.remove('active'));
        decisionTabs.querySelector('[data-section="skincare"]').classList.add('active');
        
        document.querySelectorAll('.decision-panel').forEach(panel => panel.classList.remove('active'));
        document.getElementById('panel-skincare').classList.add('active');
        
        setProgress(0);
    }

    // ==========================================================================
    // DATABASE HISTORY LOGIC
    // ==========================================================================

    function loadHistory() {
        fetch('/history')
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    renderHistoryList(data.history);
                } else {
                    console.error("Failed to load history:", data.error);
                }
            })
            .catch(error => {
                console.error("Error loading history:", error);
            });
    }

    function renderHistoryList(history) {
        const items = historyListContainer.querySelectorAll('.history-item');
        items.forEach(el => el.remove());

        if (!history || history.length === 0) {
            emptyHistoryMsg.classList.remove('hidden');
            clearHistoryBtn.style.display = 'none';
            return;
        }

        emptyHistoryMsg.classList.add('hidden');
        clearHistoryBtn.style.display = 'inline-block';

        history.forEach(scan => {
            const item = document.createElement('div');
            item.className = 'history-item animate-fade-in';
            
            let riskClass = 'risk-low';
            if (scan.risk_level.toLowerCase().includes('medium')) {
                riskClass = 'risk-medium';
            } else if (scan.risk_level.toLowerCase().includes('high')) {
                riskClass = 'risk-high';
            }

            const date = new Date(scan.timestamp);
            const dateStr = date.toLocaleDateString(undefined, {month: 'short', day: 'numeric'}) + ' ' + 
                            date.toLocaleTimeString(undefined, {hour: '2-digit', minute: '2-digit'});

            item.innerHTML = `
                <div class="history-item-thumb">
                    <img src="data:image/png;base64,${scan.original_image}" alt="Scan Thumbnail">
                </div>
                <div class="history-item-details">
                    <div class="history-item-header">
                        <span class="history-item-title">${scan.predicted_class}</span>
                        <span class="history-item-badge ${riskClass}">${scan.risk_level}</span>
                    </div>
                    <div class="history-item-meta">
                        <span>Confidence: <strong>${Math.round(scan.probability)}%</strong></span>
                        <span>${dateStr}</span>
                    </div>
                </div>
                <div class="history-item-actions">
                    <button class="delete-item-btn" title="Delete scan" data-id="${scan.id}">
                        <i class="fa-solid fa-trash-can"></i>
                    </button>
                </div>
            `;

            item.addEventListener('click', (e) => {
                if (e.target.closest('.delete-item-btn')) return;

                const compiledData = {
                    success: true,
                    demo_mode: false,
                    predicted_class: scan.predicted_class,
                    predicted_key: scan.predicted_key,
                    risk_level: scan.risk_level,
                    type: scan.type,
                    description: scan.description,
                    probability: scan.probability,
                    precautions: getPrecautionsList(scan.predicted_key),
                    skincare: getSkincareList(scan.predicted_key),
                    medical_advice: scan.medical_advice || getMedicalAdvice(scan.predicted_key),
                    images: {
                        original: scan.original_image,
                        grayscale: scan.original_image,
                        blackhat: scan.original_image,
                        hair_mask: scan.original_image,
                        hair_removed: scan.original_image,
                        enhanced: scan.enhanced_image
                    },
                    distribution: [
                        { name: scan.predicted_class, probability: Math.round(scan.probability), risk_level: scan.risk_level }
                    ]
                };

                historyDrawer.classList.remove('open');
                renderResults(compiledData);
            });

            const delBtn = item.querySelector('.delete-item-btn');
            delBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                if (confirm('Are you sure you want to delete this scan from your history?')) {
                    deleteScanItem(scan.id);
                }
            });

            historyListContainer.appendChild(item);
        });
    }

    function getPrecautionsList(key) {
        const fallbackData = {
            mel: [
                "Do NOT scratch, pick, or irritate the lesion.",
                "Avoid all direct sun exposure and tanning beds.",
                "Schedule a professional skin biopsy immediately."
            ],
            bcc: [
                "Do not attempt to squeeze, pop, or scratch the lesion.",
                "Shield the area from direct sunlight.",
                "Schedule a consultation for dermatological evaluation."
            ],
            akiec: [
                "Strictly avoid sun exposure during peak hours.",
                "Do not peel off or pick at the dry, scaly skin patches.",
                "Check for other scaly lesions on sun-exposed areas (face, scalp, ears)."
            ],
            nv: [
                "Monitor the mole monthly for changes in shape, size, border, or color.",
                "Prevent sunburns, as they increase the risk of moles mutating.",
                "Avoid friction or rubbing from tight clothing or jewelry."
            ],
            bkl: [
                "Avoid picking or scratching at the crust, which can lead to localized bleeding or infection.",
                "Prevent friction from clothes, which can make the lesion red and irritated."
            ],
            vasc: [
                "Avoid trauma or picking at the site, as vascular lesions bleed very easily.",
                "Avoid applying excessive heat or hot water to the area."
            ],
            df: [
                "Avoid squeezing or trying to pop the nodule, as it is composed of fibrous scar tissue.",
                "Be careful when shaving around the area to prevent cutting it."
            ]
        };
        return fallbackData[key] || [];
    }

    function getSkincareList(key) {
        const fallbackData = {
            mel: [
                "Keep the skin barrier hydrated with gentle, fragrance-free moisturizers.",
                "Always apply a broad-spectrum mineral sunscreen (SPF 50+) if exposed to light.",
                "Avoid chemical peels or abrasive scrubs on the affected area."
            ],
            bcc: [
                "Cleanse the skin with mild, soap-free cleansers.",
                "Maintain the skin barrier with ceramide-based moisturizers.",
                "Apply SPF 30+ sunscreen daily to prevent localized growth stimulation."
            ],
            akiec: [
                "Use thick, soothing emollients (like petrolatum or shea butter) to calm dryness.",
                "Incorporate topical antioxidants (e.g., Niacinamide) to support skin repair.",
                "Use broad-spectrum SPF 50+ sunscreen religiously before going outdoors."
            ],
            nv: [
                "Apply broad-spectrum SPF 30+ daily to all exposed skin.",
                "Keep the skin healthy with general hydration and moisture.",
                "No special clinical skincare is needed unless the mole is physically irritated."
            ],
            bkl: [
                "Soothe the area with waxy or non-comedogenic moisturizers.",
                "Avoid using harsh chemical exfoliants (like glycolic or salicylic acid) directly on the growth.",
                "Wash the area gently with a soft cloth."
            ],
            vasc: [
                "Use gentle cleansers and pat dry without rubbing.",
                "Use simple, mild lotions to keep the skin surrounding the lesion hydrated."
            ],
            df: [
                "Apply moisturizers to keep the skin smooth.",
                "Use soothing creams if the nodule feels dry or slightly itchy."
            ]
        };
        return fallbackData[key] || [];
    }

    function getMedicalAdvice(key) {
        const fallbackData = {
            mel: "CRITICAL: Seek immediate evaluation from a board-certified dermatologist. This condition requires a professional biopsy and surgical excision. Watch closely for the ABCDE rules of skin cancer.",
            bcc: "IMPORTANT: Consult a dermatologist to discuss removal options, which include surgical excision, Mohs micrographic surgery, or cryosurgery. Basal Cell Carcinoma should be professionally treated to prevent deep tissue invasion.",
            akiec: "RECOMMENDED: Visit a dermatologist for examination and treatment. Options include cryotherapy (freezing), topical chemotherapy creams (like 5-fluorouracil), or photodynamic therapy to clear the pre-cancerous cells.",
            nv: "ROUTINE: No immediate medical treatment is required. However, if the mole begins to bleed, itch, grow rapidly, or show irregular borders (asymmetry, color shifts), consult a dermatologist immediately.",
            bkl: "ROUTINE: These growths are entirely harmless and do not turn into cancer. If they become irritated, catch on clothing, or are a cosmetic concern, a dermatologist can quickly remove them using cryotherapy or light curettage.",
            vasc: "ROUTINE: Generally harmless. If the lesion bleeds excessively, becomes painful, or grows rapidly (which can happen with pyogenic granulomas), seek clinical evaluation for treatment, which may include laser therapy or electrocautery.",
            df: "ROUTINE: Dermatofibromas are benign and do not require treatment. If they are painful, itchy, or cosmetically undesirable, they can be surgically removed by a dermatologist, though this leaves a small scar."
        };
        return fallbackData[key] || "Routine monitoring recommended.";
    }

    function deleteScanItem(id) {
        fetch(`/history/delete/${id}`, { method: 'POST' })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    loadHistory();
                } else {
                    alert('Failed to delete scan: ' + data.error);
                }
            })
            .catch(error => {
                console.error("Error deleting scan:", error);
            });
    }

    function clearHistory() {
        fetch('/history/clear', { method: 'POST' })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    loadHistory();
                } else {
                    alert('Failed to clear history: ' + data.error);
                }
            })
            .catch(error => {
                console.error("Error clearing history:", error);
            });
    }

    // Load history on initial start
    loadHistory();
});
