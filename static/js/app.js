/**
 * Network Traffic Classifier - Frontend Application
 */

// State
let selectedModel = 'LightGBM';

// Demo data for each traffic class - Extracted from real ISCX dataset
// These are actual samples from the dataset for accurate classification
const demoData = {
    'BROWSING': {
        duration: 204192.0,
        total_fiat: 0,
        total_biat: 0,
        min_fiat: 27596.0,
        min_biat: 65851.0,
        max_fiat: 7698.5555555556,
        max_biat: 8118.4090909091,
        mean_fiat: 11952.8074463564,
        mean_biat: 16638.2382971936,
        flowPktsPerSecond: 205.6887635167,
        flowBytesPerSecond: 72211.4480488952,
        min_flowiat: 14.0,
        max_flowiat: 65618.0,
        mean_flowiat: 4980.2926829268,
        std_flowiat: 12805.3520329273,
        min_active: 0,
        mean_active: 0.0,
        max_active: 0,
        std_active: 0.0,
        min_idle: 0,
        mean_idle: 0.0,
        max_idle: 0,
        std_idle: 0.0
    },
    'CHAT': {
        duration: 170173.0,
        total_fiat: 0,
        total_biat: 0,
        min_fiat: 0,
        min_biat: 0,
        max_fiat: 0.0,
        max_biat: 0.0,
        mean_fiat: 0.0,
        mean_biat: 0.0,
        flowPktsPerSecond: 11.7527457352,
        flowBytesPerSecond: 505.3680666146,
        min_flowiat: 170173.0,
        max_flowiat: 170173.0,
        mean_flowiat: 170173.0,
        std_flowiat: 0.0,
        min_active: 0,
        mean_active: 0.0,
        max_active: 0,
        std_active: 0.0,
        min_idle: 0,
        mean_idle: 0.0,
        max_idle: 0,
        std_idle: 0.0
    },
    'FILE_TRANSFER': {
        duration: 14768538.0,
        total_fiat: 0,
        total_biat: 0,
        min_fiat: 1217379.0,
        min_biat: 1009725.0,
        max_fiat: 246142.3,
        max_biat: 380546.710526316,
        mean_fiat: 298382.501314513,
        mean_biat: 256942.618599626,
        flowPktsPerSecond: 6.7711509426,
        flowBytesPerSecond: 1437.9216141774,
        min_flowiat: 17.0,
        max_flowiat: 747323.0,
        mean_flowiat: 149177.151515152,
        std_flowiat: 166262.148031407,
        min_active: 0,
        mean_active: 0.0,
        max_active: 0,
        std_active: 0.0,
        min_idle: 0,
        mean_idle: 0.0,
        max_idle: 0,
        std_idle: 0.0
    },
    'MAIL': {
        duration: 116859081.0,
        total_fiat: 0,
        total_biat: 0,
        min_fiat: 20024520.0,
        min_biat: 20001201.0,
        max_fiat: 564461.096618358,
        max_biat: 950072.25203252,
        mean_fiat: 2342946.59622368,
        mean_biat: 2979525.83388674,
        flowPktsPerSecond: 2.8410286745,
        flowBytesPerSecond: 220.4963429415,
        min_flowiat: 16.0,
        max_flowiat: 19999738.0,
        mean_flowiat: 353048.583081571,
        std_flowiat: 1867476.13273464,
        min_active: 6384393.0,
        mean_active: 11400000.0,
        max_active: 21453354.0,
        std_active: 4684503.83269093,
        min_idle: 5001750.0,
        mean_idle: 9970629.9,
        max_idle: 19999738.0,
        std_idle: 4653120.96721358
    },
    'P2P': {
        duration: 89068.0,
        total_fiat: 0,
        total_biat: 0,
        min_fiat: 0,
        min_biat: 0,
        max_fiat: 0,
        max_biat: 0,
        mean_fiat: 0.0,
        mean_biat: 0.0,
        flowPktsPerSecond: 22.4547536714,
        flowBytesPerSecond: 1032.9186688822,
        min_flowiat: 89068.0,
        max_flowiat: 89068.0,
        mean_flowiat: 89068.0,
        std_flowiat: 0.0,
        min_active: 0,
        mean_active: 0.0,
        max_active: 0,
        std_active: 0.0,
        min_idle: 0,
        mean_idle: 0.0,
        max_idle: 0,
        std_idle: 0.0
    },
    'STREAMING': {
        duration: 12514828.0,
        total_fiat: 0,
        total_biat: 0,
        min_fiat: 3226144.0,
        min_biat: 3371369.0,
        max_fiat: 5678.2341197822,
        max_biat: 2865.2986078886,
        mean_fiat: 106888.956636816,
        mean_biat: 80651.2676123401,
        flowPktsPerSecond: 520.6623694708,
        flowBytesPerSecond: 477549.27195164,
        min_flowiat: 1.0,
        max_flowiat: 3226144.0,
        mean_flowiat: 1920.9252494244,
        std_flowiat: 62214.7959828713,
        min_active: 1205713.0,
        mean_active: 3003353.5,
        max_active: 4005199.0,
        std_active: 1326688.72391806,
        min_idle: 1044811.0,
        mean_idle: 2344151.25,
        max_idle: 3226144.0,
        std_idle: 1009645.4072029
    },
    'VOIP': {
        duration: 8272911.0,
        total_fiat: 0,
        total_biat: 0,
        min_fiat: 1030079.0,
        min_biat: 1029669.0,
        max_fiat: 459606.166666667,
        max_biat: 288368.0,
        mean_fiat: 469192.702856695,
        mean_biat: 457870.159057533,
        flowPktsPerSecond: 5.8020689453,
        flowBytesPerSecond: 371.8159182421,
        min_flowiat: 1.0,
        max_flowiat: 1029669.0,
        mean_flowiat: 176019.382978723,
        std_flowiat: 360925.668878983,
        min_active: 1002717.0,
        mean_active: 1213655.6,
        max_active: 2003627.0,
        std_active: 441785.036352014,
        min_idle: 1001616.0,
        mean_idle: 1012422.4,
        max_idle: 1029669.0,
        std_idle: 14330.3368435637
    }
};

// Class info
const classInfo = {
    'BROWSING': { icon: '🌐', desc: 'Web Browsing Traffic', color: '#3498db' },
    'CHAT': { icon: '💬', desc: 'Chat/Messaging Apps', color: '#9b59b6' },
    'FILE_TRANSFER': { icon: '📁', desc: 'File Transfer (FTP, etc.)', color: '#e67e22' },
    'MAIL': { icon: '📧', desc: 'Email Traffic', color: '#1abc9c' },
    'P2P': { icon: '🔗', desc: 'Peer-to-Peer Traffic', color: '#e74c3c' },
    'STREAMING': { icon: '🎬', desc: 'Video/Audio Streaming', color: '#f39c12' },
    'VOIP': { icon: '📞', desc: 'Voice over IP', color: '#2ecc71' }
};

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initModelSelection();
    initTabs();
    initPredictForm();
    initCSVUpload();
    initDemoButtons();
    initRandomFill();
});

// Model Selection
function initModelSelection() {
    const modelBtns = document.querySelectorAll('.model-btn');
    modelBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            modelBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            selectedModel = btn.dataset.model;
        });
    });
}

// Tabs
function initTabs() {
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const tabId = btn.dataset.tab;
            
            tabBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));
            
            btn.classList.add('active');
            document.getElementById(`${tabId}-tab`).classList.add('active');
        });
    });
}

// Predict Form
function initPredictForm() {
    const form = document.getElementById('predict-form');
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const formData = new FormData(form);
        const features = {};
        for (let [key, value] of formData.entries()) {
            features[key] = parseFloat(value) || 0;
        }
        
        await predict(features);
    });
}

// CSV Upload
function initCSVUpload() {
    const uploadZone = document.getElementById('upload-zone');
    const fileInput = document.getElementById('csv-file');
    const fileInfo = document.getElementById('file-info');
    const predictBtn = document.getElementById('csv-predict-btn');
    
    uploadZone.addEventListener('click', () => fileInput.click());
    
    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('dragover');
    });
    
    uploadZone.addEventListener('dragleave', () => {
        uploadZone.classList.remove('dragover');
    });
    
    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });
    
    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
            handleFile(fileInput.files[0]);
        }
    });
    
    function handleFile(file) {
        if (!file.name.endsWith('.csv')) {
            alert('Please upload a CSV file');
            return;
        }
        
        fileInfo.style.display = 'flex';
        fileInfo.querySelector('.file-name').textContent = file.name;
        predictBtn.disabled = false;
        
        fileInfo.querySelector('.remove-file').onclick = () => {
            fileInput.value = '';
            fileInfo.style.display = 'none';
            predictBtn.disabled = true;
        };
    }
    
    predictBtn.addEventListener('click', async () => {
        if (fileInput.files.length === 0) return;
        
        const formData = new FormData();
        formData.append('file', fileInput.files[0]);
        formData.append('model', selectedModel);
        
        predictBtn.classList.add('loading');
        predictBtn.disabled = true;
        
        try {
            const response = await fetch('/predict_csv', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (data.success) {
                showCSVResults(data);
            } else {
                alert('Error: ' + data.error);
            }
        } catch (error) {
            alert('Error: ' + error.message);
        } finally {
            predictBtn.classList.remove('loading');
            predictBtn.disabled = false;
        }
    });
}

// Demo Buttons - Show QoS directly without prediction
function initDemoButtons() {
    const demoBtns = document.querySelectorAll('.demo-btn');
    demoBtns.forEach(btn => {
        btn.addEventListener('click', async () => {
            const className = btn.dataset.class;
            
            // Show demo result directly (no prediction needed)
            showDemoResult(className);
        });
    });
}

// Show demo result directly for a class
async function showDemoResult(className) {
    const resultsSection = document.getElementById('results');
    const csvResults = document.getElementById('csv-results');
    const qosResults = document.getElementById('qos-results');
    
    csvResults.style.display = 'none';
    resultsSection.style.display = 'block';
    qosResults.style.display = 'block';
    
    const info = classInfo[className] || { icon: '❓', desc: 'Unknown', color: '#666' };
    
    document.getElementById('result-icon').textContent = info.icon;
    document.getElementById('result-class').textContent = className;
    document.getElementById('result-desc').textContent = info.desc;
    document.getElementById('result-model').textContent = 'Demo Mode';
    
    // Show 100% probability for selected class
    const probContainer = document.getElementById('probabilities');
    probContainer.innerHTML = '';
    
    for (let label of Object.keys(classInfo)) {
        const prob = label === className ? 100 : 0;
        const probBar = document.createElement('div');
        probBar.className = 'prob-bar';
        probBar.innerHTML = `
            <span class="prob-label">${classInfo[label]?.icon || ''} ${label}</span>
            <div class="prob-track">
                <div class="prob-fill" style="width: ${prob}%">
                    ${prob > 0 ? `<span class="prob-value">${prob.toFixed(1)}%</span>` : ''}
                </div>
            </div>
        `;
        probContainer.appendChild(probBar);
    }
    
    // Fetch QoS data from server
    try {
        const response = await fetch(`/qos/${className}`);
        const qos = await response.json();
        showQoSRecommendations(qos);
    } catch (error) {
        console.error('Error fetching QoS:', error);
    }
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

// Predict function
async function predict(features) {
    const predictBtn = document.querySelector('#predict-form .predict-btn');
    predictBtn.classList.add('loading');
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                model: selectedModel,
                features: features
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            showResult(data);
        } else {
            alert('Error: ' + data.error);
        }
    } catch (error) {
        alert('Error: ' + error.message);
    } finally {
        predictBtn.classList.remove('loading');
    }
}

// Show result
function showResult(data) {
    const resultsSection = document.getElementById('results');
    const csvResults = document.getElementById('csv-results');
    const qosResults = document.getElementById('qos-results');
    const perfResults = document.getElementById('performance-results');
    
    csvResults.style.display = 'none';
    resultsSection.style.display = 'block';
    qosResults.style.display = 'block';
    if (perfResults) perfResults.style.display = 'block';
    
    // Get app type for icon/color (VPN-VOIP -> VOIP for lookup)
    const appType = data.app_type || data.prediction;
    const info = classInfo[appType] || { icon: '❓', desc: 'Unknown', color: '#666' };
    
    // Show VPN badge if detected
    let classDisplay = data.prediction;
    let descDisplay = info.desc;
    
    if (data.is_vpn === true) {
        descDisplay = `🔒 VPN Traffic - ${info.desc}`;
    } else if (data.is_vpn === false) {
        descDisplay = `🌐 Non-VPN - ${info.desc}`;
    }
    
    document.getElementById('result-icon').textContent = info.icon;
    document.getElementById('result-class').textContent = classDisplay;
    document.getElementById('result-desc').textContent = descDisplay;
    document.getElementById('result-model').textContent = data.model;
    
    // Show VPN confidence if available
    if (data.vpn_confidence !== null && data.vpn_confidence !== undefined) {
        const vpnBadge = document.getElementById('vpn-badge');
        if (vpnBadge) {
            vpnBadge.style.display = 'inline-block';
            vpnBadge.textContent = data.is_vpn ? 
                `🔒 VPN (${data.vpn_confidence.toFixed(1)}%)` : 
                `🌐 Non-VPN (${data.vpn_confidence.toFixed(1)}%)`;
            vpnBadge.className = data.is_vpn ? 'vpn-badge vpn' : 'vpn-badge non-vpn';
        }
    }
    
    // Show probabilities
    const probContainer = document.getElementById('probabilities');
    probContainer.innerHTML = '';
    
    if (data.probabilities && Object.keys(data.probabilities).length > 0) {
        // Sort by probability
        const sorted = Object.entries(data.probabilities)
            .sort((a, b) => b[1] - a[1]);
        
        for (let [label, prob] of sorted) {
            const probBar = document.createElement('div');
            probBar.className = 'prob-bar';
            probBar.innerHTML = `
                <span class="prob-label">${classInfo[label]?.icon || ''} ${label}</span>
                <div class="prob-track">
                    <div class="prob-fill" style="width: ${prob}%">
                        <span class="prob-value">${prob.toFixed(1)}%</span>
                    </div>
                </div>
            `;
            probContainer.appendChild(probBar);
        }
    }
    
    // Show QoS recommendations
    if (data.qos) {
        showQoSRecommendations(data.qos);
    }
    
    // Show performance estimates
    if (data.performance) {
        showPerformanceEstimates(data.performance);
    }
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

// Show QoS Recommendations
function showQoSRecommendations(qos) {
    // Priority
    const priorityValue = document.getElementById('qos-priority-value');
    priorityValue.textContent = qos.priority?.name || 'UNKNOWN';
    priorityValue.style.color = qos.priority?.color || '#666';
    
    // Priority bar
    const priorityBar = document.getElementById('priority-bar');
    priorityBar.innerHTML = '';
    const priorityLevel = qos.priority?.level || 5;
    const priorityClass = `priority-${qos.priority?.name?.toLowerCase() || 'best-effort'}`;
    
    for (let i = 1; i <= 5; i++) {
        const segment = document.createElement('div');
        segment.className = 'priority-segment';
        if (i <= (6 - priorityLevel)) {
            segment.classList.add('active', priorityClass);
        }
        priorityBar.appendChild(segment);
    }
    
    // Bandwidth
    document.getElementById('qos-bandwidth').textContent = 
        `${qos.bandwidth?.min_display || '?'} - ${qos.bandwidth?.max_display || '?'}`;
    
    // Latency
    document.getElementById('qos-latency').textContent = qos.latency?.display || '?';
    
    // Jitter
    document.getElementById('qos-jitter').textContent = qos.jitter?.display || '?';
    
    // Packet Loss
    document.getElementById('qos-packet-loss').textContent = qos.packet_loss?.display || '?';
    
    // DSCP
    document.getElementById('qos-dscp').textContent = 
        `${qos.dscp?.value || 0} (${qos.dscp?.hex || '0x00'})`;
    
    // Actions
    const actionsList = document.getElementById('qos-actions-list');
    actionsList.innerHTML = '';
    if (qos.actions && qos.actions.length > 0) {
        for (let action of qos.actions) {
            const li = document.createElement('li');
            li.textContent = action;
            actionsList.appendChild(li);
        }
    }
    
    // Description
    document.getElementById('qos-description').textContent = qos.description || '';
}

// Show Performance Estimates
function showPerformanceEstimates(perf) {
    const perfSection = document.getElementById('performance-results');
    if (!perfSection) return;
    
    // Quality score and circle
    const qualityScore = document.getElementById('quality-score');
    const qualityLevel = document.getElementById('quality-level');
    const qualityCircle = document.getElementById('quality-circle');
    
    if (qualityScore) qualityScore.textContent = perf.quality_score;
    if (qualityLevel) qualityLevel.textContent = perf.quality_level;
    if (qualityCircle) {
        qualityCircle.style.setProperty('--quality-percent', perf.quality_score);
        qualityCircle.style.setProperty('--quality-color', perf.quality_color);
    }
    
    // Delay indicator
    const perfDelay = document.getElementById('perf-delay');
    if (perfDelay) {
        if (perf.delay_indicator_ms > 1000) {
            perfDelay.textContent = `~${(perf.delay_indicator_ms / 1000).toFixed(1)}s`;
        } else {
            perfDelay.textContent = `~${perf.delay_indicator_ms.toFixed(1)}ms`;
        }
    }
    
    // Jitter indicator
    const perfJitter = document.getElementById('perf-jitter');
    if (perfJitter) {
        perfJitter.textContent = `~${perf.jitter_indicator_ms.toFixed(1)}ms`;
    }
    
    // Throughput
    const perfThroughput = document.getElementById('perf-throughput');
    if (perfThroughput) {
        if (perf.bytes_per_sec > 1000000) {
            perfThroughput.textContent = `${(perf.bytes_per_sec / 1000000).toFixed(2)} MB/s`;
        } else if (perf.bytes_per_sec > 1000) {
            perfThroughput.textContent = `${(perf.bytes_per_sec / 1000).toFixed(2)} KB/s`;
        } else {
            perfThroughput.textContent = `${perf.bytes_per_sec.toFixed(0)} B/s`;
        }
    }
}

// Show CSV results
function showCSVResults(data) {
    const resultsSection = document.getElementById('results');
    const csvResults = document.getElementById('csv-results');
    
    resultsSection.style.display = 'none';
    csvResults.style.display = 'block';
    
    // Summary
    const summaryContainer = document.getElementById('csv-summary');
    summaryContainer.innerHTML = '';
    
    for (let [label, count] of Object.entries(data.summary)) {
        const info = classInfo[label] || { icon: '❓' };
        const item = document.createElement('div');
        item.className = 'summary-item';
        item.innerHTML = `
            <span class="summary-icon">${info.icon}</span>
            <div>
                <div class="summary-count">${count}</div>
                <div class="summary-label">${label}</div>
            </div>
        `;
        summaryContainer.appendChild(item);
    }
    
    // Table
    const tbody = document.querySelector('#csv-table tbody');
    tbody.innerHTML = '';
    
    for (let result of data.results) {
        const info = classInfo[result.prediction] || { icon: '❓', desc: 'Unknown' };
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${result.row}</td>
            <td><span style="font-size: 1.2rem">${info.icon}</span> ${result.prediction}</td>
            <td style="color: var(--text-muted)">${info.desc}</td>
        `;
        tbody.appendChild(row);
    }
    
    if (data.total_rows > 100) {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td colspan="3" style="text-align: center; color: var(--text-muted)">
                ... and ${data.total_rows - 100} more rows
            </td>
        `;
        tbody.appendChild(row);
    }
    
    csvResults.scrollIntoView({ behavior: 'smooth' });
}

// Random Fill functionality
function initRandomFill() {
    const randomBtn = document.getElementById('random-fill-btn');
    if (!randomBtn) return;
    
    randomBtn.addEventListener('click', fillRandomValues);
}

// Generate random values based on realistic network traffic ranges
function fillRandomValues() {
    // Random ranges based on actual dataset statistics
    const ranges = {
        duration: { min: 10000, max: 120000000 },           // microseconds
        flowPktsPerSecond: { min: 1, max: 500 },
        flowBytesPerSecond: { min: 100, max: 500000 },
        min_fiat: { min: 0, max: 100000 },
        max_fiat: { min: 100000, max: 30000000 },
        mean_fiat: { min: 1000, max: 5000000 },
        min_biat: { min: 0, max: 100000 },
        max_biat: { min: 100000, max: 30000000 },
        mean_biat: { min: 1000, max: 5000000 },
        min_flowiat: { min: 1, max: 10000 },
        max_flowiat: { min: 10000, max: 30000000 },
        mean_flowiat: { min: 1000, max: 5000000 },
        std_flowiat: { min: 0, max: 1000000 },
        min_active: { min: 0, max: 100000 },
        max_active: { min: 0, max: 1000000 },
        mean_active: { min: 0, max: 500000 },
        std_active: { min: 0, max: 200000 },
        min_idle: { min: 0, max: 1000000 },
        max_idle: { min: 0, max: 10000000 },
        mean_idle: { min: 0, max: 5000000 },
        std_idle: { min: 0, max: 2000000 }
    };
    
    // Fill form with random values
    const form = document.getElementById('predict-form');
    
    for (const [field, range] of Object.entries(ranges)) {
        const input = form.querySelector(`[name="${field}"]`);
        if (input) {
            // Generate random value with some randomness in distribution
            let value;
            if (Math.random() < 0.3) {
                // 30% chance of lower range value
                value = range.min + Math.random() * (range.max - range.min) * 0.2;
            } else if (Math.random() < 0.7) {
                // 40% chance of mid range value
                value = range.min + (range.max - range.min) * 0.2 + 
                        Math.random() * (range.max - range.min) * 0.6;
            } else {
                // 30% chance of higher range value
                value = range.min + (range.max - range.min) * 0.8 + 
                        Math.random() * (range.max - range.min) * 0.2;
            }
            
            // Round to reasonable precision
            if (value > 1000) {
                value = Math.round(value);
            } else {
                value = Math.round(value * 100) / 100;
            }
            
            input.value = value;
        }
    }
    
    // Set total_fiat and total_biat to 0 (as per model expectation)
    const totalFiat = form.querySelector('[name="total_fiat"]');
    const totalBiat = form.querySelector('[name="total_biat"]');
    if (totalFiat) totalFiat.value = 0;
    if (totalBiat) totalBiat.value = 0;
    
    // Visual feedback
    const btn = document.getElementById('random-fill-btn');
    const originalText = btn.innerHTML;
    btn.innerHTML = '✅ Filled!';
    btn.style.background = 'linear-gradient(135deg, #10b981, #059669)';
    
    setTimeout(() => {
        btn.innerHTML = originalText;
        btn.style.background = '';
    }, 1000);
}

