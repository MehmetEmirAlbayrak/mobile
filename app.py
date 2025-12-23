"""
Mobile Network Traffic Classification - Web Application
Flask backend for serving ML models
"""

from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.qos import qos_optimizer, get_qos_for_traffic, allocate_bandwidth_for_traffic

app = Flask(__name__)

# Load trained models and preprocessing artifacts
MODELS_DIR = 'models'
models = {}
scaler = None
label_encoder = None
feature_columns = None

# VPN Detector artifacts
vpn_detector = None
vpn_scaler = None
vpn_feature_columns = None

def load_models():
    """Load all available trained models and preprocessing artifacts"""
    global models, scaler, label_encoder, feature_columns
    global vpn_detector, vpn_scaler, vpn_feature_columns
    
    model_files = {
        'LightGBM': 'LightGBM.pkl',
        'RandomForest': 'RandomForest.pkl',
        'XGBoost': 'XGBoost.pkl'
    }
    
    for name, filename in model_files.items():
        path = os.path.join(MODELS_DIR, filename)
        if os.path.exists(path):
            try:
                models[name] = joblib.load(path)
                print(f"✓ Loaded {name}")
            except Exception as e:
                print(f"✗ Failed to load {name}: {e}")
    
    # Load preprocessing artifacts if available
    scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        print("✓ Loaded scaler")
    else:
        print("⚠ Scaler not found - predictions may be less accurate")
    
    le_path = os.path.join(MODELS_DIR, 'label_encoder.pkl')
    if os.path.exists(le_path):
        label_encoder = joblib.load(le_path)
        print("✓ Loaded label encoder")
    
    fc_path = os.path.join(MODELS_DIR, 'feature_columns.pkl')
    if os.path.exists(fc_path):
        feature_columns = joblib.load(fc_path)
        print(f"✓ Loaded feature columns ({len(feature_columns)} features)")
    
    # Load VPN Detector
    vpn_path = os.path.join(MODELS_DIR, 'VPN_Detector.pkl')
    if os.path.exists(vpn_path):
        vpn_detector = joblib.load(vpn_path)
        print("✓ Loaded VPN Detector")
    
    vpn_scaler_path = os.path.join(MODELS_DIR, 'vpn_scaler.pkl')
    if os.path.exists(vpn_scaler_path):
        vpn_scaler = joblib.load(vpn_scaler_path)
        print("✓ Loaded VPN scaler")
    
    vpn_fc_path = os.path.join(MODELS_DIR, 'vpn_feature_columns.pkl')
    if os.path.exists(vpn_fc_path):
        vpn_feature_columns = joblib.load(vpn_fc_path)
        print(f"✓ Loaded VPN feature columns ({len(vpn_feature_columns)} features)")
    
    print(f"\n{len(models)} models loaded successfully!")
    if vpn_detector:
        print("VPN Detector loaded - Two-stage classification enabled!")

# Base feature names (user inputs these)
BASE_FEATURES = [
    'duration', 'total_fiat', 'total_biat', 'min_fiat', 'min_biat',
    'max_fiat', 'max_biat', 'mean_fiat', 'mean_biat', 'flowPktsPerSecond',
    'flowBytesPerSecond', 'min_flowiat', 'max_flowiat', 'mean_flowiat',
    'std_flowiat', 'min_active', 'mean_active', 'max_active', 'std_active',
    'min_idle', 'mean_idle', 'max_idle', 'std_idle'
]

# All 38 features after engineering (must match training)
FEATURE_NAMES = [
    'std_active', 'duration', 'std_flowiat', 'min_fiat', 'max_biat', 
    'min_biat', 'mean_biat', 'mean_fiat', 'mean_flowiat', 'max_flowiat', 
    'min_idle', 'std_idle', 'max_idle', 'flowBytesPerSecond', 'flowPktsPerSecond', 
    'mean_idle', 'max_fiat', 'mean_active', 'max_active', 'min_active', 
    'min_flowiat', 'flowiat_cov', 'active_cov', 'idle_cov', 'active_idle_ratio', 
    'bytes_per_pkt', 'fiat_range', 'biat_range', 'flowiat_range', 'active_range', 
    'idle_range', 'fiat_skew_approx', 'biat_skew_approx', 'flowiat_skew_approx', 
    'active_skew_approx', 'idle_skew_approx', 'duration_bytes_interaction', 
    'duration_pkts_interaction'
]

# Class labels
CLASS_LABELS = ['BROWSING', 'CHAT', 'FILE_TRANSFER', 'MAIL', 'P2P', 'STREAMING', 'VOIP']

# Class descriptions and icons
CLASS_INFO = {
    'BROWSING': {'icon': '🌐', 'desc': 'Web Browsing Traffic', 'color': '#3498db'},
    'CHAT': {'icon': '💬', 'desc': 'Chat/Messaging Apps', 'color': '#9b59b6'},
    'FILE_TRANSFER': {'icon': '📁', 'desc': 'File Transfer (FTP, etc.)', 'color': '#e67e22'},
    'MAIL': {'icon': '📧', 'desc': 'Email Traffic', 'color': '#1abc9c'},
    'P2P': {'icon': '🔗', 'desc': 'Peer-to-Peer Traffic', 'color': '#e74c3c'},
    'STREAMING': {'icon': '🎬', 'desc': 'Video/Audio Streaming', 'color': '#f39c12'},
    'VOIP': {'icon': '📞', 'desc': 'Voice over IP', 'color': '#2ecc71'}
}

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html', 
                         models=list(models.keys()),
                         classes=CLASS_INFO,
                         features=BASE_FEATURES)

def engineer_features(features):
    """
    Apply the same feature engineering as in training.
    Uses feature_columns from training to ensure correct order.
    """
    # Get base values with defaults
    duration = float(features.get('duration', 0))
    total_fiat = float(features.get('total_fiat', 0))
    total_biat = float(features.get('total_biat', 0))
    min_fiat = float(features.get('min_fiat', 0))
    min_biat = float(features.get('min_biat', 0))
    max_fiat = float(features.get('max_fiat', 0))
    max_biat = float(features.get('max_biat', 0))
    mean_fiat = float(features.get('mean_fiat', 0))
    mean_biat = float(features.get('mean_biat', 0))
    flowPktsPerSecond = float(features.get('flowPktsPerSecond', 0))
    flowBytesPerSecond = float(features.get('flowBytesPerSecond', 0))
    min_flowiat = float(features.get('min_flowiat', 0))
    max_flowiat = float(features.get('max_flowiat', 0))
    mean_flowiat = float(features.get('mean_flowiat', 0))
    std_flowiat = float(features.get('std_flowiat', 0))
    min_active = float(features.get('min_active', 0))
    mean_active = float(features.get('mean_active', 0))
    max_active = float(features.get('max_active', 0))
    std_active = float(features.get('std_active', 0))
    min_idle = float(features.get('min_idle', 0))
    mean_idle = float(features.get('mean_idle', 0))
    max_idle = float(features.get('max_idle', 0))
    std_idle = float(features.get('std_idle', 0))
    
    # Engineered features
    # COV (Coefficient of Variation)
    flowiat_cov = std_flowiat / (mean_flowiat + 1e-6)
    active_cov = std_active / (mean_active + 1e-6)
    idle_cov = std_idle / (mean_idle + 1e-6)
    
    # Ratios
    active_idle_ratio = mean_active / (mean_idle + 1e-6)
    bytes_per_pkt = flowBytesPerSecond / (flowPktsPerSecond + 1e-6)
    
    # Range features
    fiat_range = max_fiat - min_fiat
    biat_range = max_biat - min_biat
    flowiat_range = max_flowiat - min_flowiat
    active_range = max_active - min_active
    idle_range = max_idle - min_idle
    
    # Skewness approximation
    fiat_skew_approx = (max_fiat - mean_fiat) / (mean_fiat - min_fiat + 1e-6)
    biat_skew_approx = (max_biat - mean_biat) / (mean_biat - min_biat + 1e-6)
    flowiat_skew_approx = (max_flowiat - mean_flowiat) / (mean_flowiat - min_flowiat + 1e-6)
    active_skew_approx = (max_active - mean_active) / (mean_active - min_active + 1e-6)
    idle_skew_approx = (max_idle - mean_idle) / (mean_idle - min_idle + 1e-6)
    
    # Interaction features
    duration_bytes_interaction = duration * flowBytesPerSecond
    duration_pkts_interaction = duration * flowPktsPerSecond
    
    # Create a dictionary of all computed features
    all_features = {
        'duration': duration,
        'total_fiat': total_fiat,
        'total_biat': total_biat,
        'min_fiat': min_fiat,
        'min_biat': min_biat,
        'max_fiat': max_fiat,
        'max_biat': max_biat,
        'mean_fiat': mean_fiat,
        'mean_biat': mean_biat,
        'flowPktsPerSecond': flowPktsPerSecond,
        'flowBytesPerSecond': flowBytesPerSecond,
        'min_flowiat': min_flowiat,
        'max_flowiat': max_flowiat,
        'mean_flowiat': mean_flowiat,
        'std_flowiat': std_flowiat,
        'min_active': min_active,
        'mean_active': mean_active,
        'max_active': max_active,
        'std_active': std_active,
        'min_idle': min_idle,
        'mean_idle': mean_idle,
        'max_idle': max_idle,
        'std_idle': std_idle,
        'flowiat_cov': flowiat_cov,
        'active_cov': active_cov,
        'idle_cov': idle_cov,
        'active_idle_ratio': active_idle_ratio,
        'bytes_per_pkt': bytes_per_pkt,
        'fiat_range': fiat_range,
        'biat_range': biat_range,
        'flowiat_range': flowiat_range,
        'active_range': active_range,
        'idle_range': idle_range,
        'fiat_skew_approx': fiat_skew_approx,
        'biat_skew_approx': biat_skew_approx,
        'flowiat_skew_approx': flowiat_skew_approx,
        'active_skew_approx': active_skew_approx,
        'idle_skew_approx': idle_skew_approx,
        'duration_bytes_interaction': duration_bytes_interaction,
        'duration_pkts_interaction': duration_pkts_interaction
    }
    
    # Build feature vector in the EXACT order from training
    if feature_columns is not None:
        feature_vector = [all_features.get(col, 0) for col in feature_columns]
    else:
        # Fallback to hardcoded order if feature_columns not loaded
        feature_vector = list(all_features.values())
    
    return feature_vector

def estimate_performance(features):
    """
    Estimate network performance indicators from flow features.
    This provides insights about the current traffic quality.
    """
    mean_flowiat = float(features.get('mean_flowiat', 0))
    std_flowiat = float(features.get('std_flowiat', 0))
    flowPktsPerSecond = float(features.get('flowPktsPerSecond', 0))
    flowBytesPerSecond = float(features.get('flowBytesPerSecond', 0))
    mean_active = float(features.get('mean_active', 0))
    mean_idle = float(features.get('mean_idle', 0))
    
    # Convert microseconds to milliseconds for readability
    delay_indicator_ms = mean_flowiat / 1000 if mean_flowiat > 0 else 0
    jitter_indicator_ms = std_flowiat / 1000 if std_flowiat > 0 else 0
    
    # Calculate quality score (0-100)
    # Lower delay and jitter = higher quality
    quality_score = 100
    
    # Penalize high delay (> 100ms is bad for real-time)
    if delay_indicator_ms > 500:
        quality_score -= 40
    elif delay_indicator_ms > 200:
        quality_score -= 25
    elif delay_indicator_ms > 100:
        quality_score -= 15
    elif delay_indicator_ms > 50:
        quality_score -= 5
    
    # Penalize high jitter (> 30ms is bad for VoIP)
    if jitter_indicator_ms > 100:
        quality_score -= 30
    elif jitter_indicator_ms > 50:
        quality_score -= 20
    elif jitter_indicator_ms > 30:
        quality_score -= 10
    elif jitter_indicator_ms > 10:
        quality_score -= 5
    
    # Boost for good packet rate
    if flowPktsPerSecond > 100:
        quality_score = min(100, quality_score + 5)
    
    quality_score = max(0, min(100, quality_score))
    
    # Determine quality level
    if quality_score >= 80:
        quality_level = "Excellent"
        quality_color = "#10b981"  # green
    elif quality_score >= 60:
        quality_level = "Good"
        quality_color = "#22c55e"  # light green
    elif quality_score >= 40:
        quality_level = "Fair"
        quality_color = "#f59e0b"  # yellow
    elif quality_score >= 20:
        quality_level = "Poor"
        quality_color = "#f97316"  # orange
    else:
        quality_level = "Bad"
        quality_color = "#ef4444"  # red
    
    # Utilization estimate
    if mean_active > 0 and mean_idle > 0:
        utilization = (mean_active / (mean_active + mean_idle)) * 100
    else:
        utilization = 50  # Unknown
    
    return {
        'delay_indicator_ms': round(delay_indicator_ms, 2),
        'jitter_indicator_ms': round(jitter_indicator_ms, 2),
        'quality_score': round(quality_score),
        'quality_level': quality_level,
        'quality_color': quality_color,
        'packets_per_sec': round(flowPktsPerSecond, 2),
        'bytes_per_sec': round(flowBytesPerSecond, 2),
        'utilization_percent': round(utilization, 1)
    }

def detect_vpn(features):
    """
    Stage 1: Detect if traffic is VPN or not.
    Returns: (is_vpn: bool, vpn_confidence: float)
    """
    if vpn_detector is None or vpn_scaler is None or vpn_feature_columns is None:
        return None, 0.0
    
    # Build feature vector for VPN detection
    all_features = build_all_features(features)
    feature_vector = [all_features.get(col, 0) for col in vpn_feature_columns]
    
    X = np.array(feature_vector).reshape(1, -1)
    X = np.log1p(X)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = vpn_scaler.transform(X)
    
    prediction = vpn_detector.predict(X)[0]
    proba = vpn_detector.predict_proba(X)[0]
    
    is_vpn = bool(prediction == 1)
    confidence = float(proba[1] if is_vpn else proba[0]) * 100
    
    return is_vpn, confidence

def build_all_features(features):
    """Build all engineered features from base features"""
    duration = float(features.get('duration', 0))
    total_fiat = float(features.get('total_fiat', 0))
    total_biat = float(features.get('total_biat', 0))
    min_fiat = float(features.get('min_fiat', 0))
    min_biat = float(features.get('min_biat', 0))
    max_fiat = float(features.get('max_fiat', 0))
    max_biat = float(features.get('max_biat', 0))
    mean_fiat = float(features.get('mean_fiat', 0))
    mean_biat = float(features.get('mean_biat', 0))
    flowPktsPerSecond = float(features.get('flowPktsPerSecond', 0))
    flowBytesPerSecond = float(features.get('flowBytesPerSecond', 0))
    min_flowiat = float(features.get('min_flowiat', 0))
    max_flowiat = float(features.get('max_flowiat', 0))
    mean_flowiat = float(features.get('mean_flowiat', 0))
    std_flowiat = float(features.get('std_flowiat', 0))
    min_active = float(features.get('min_active', 0))
    mean_active = float(features.get('mean_active', 0))
    max_active = float(features.get('max_active', 0))
    std_active = float(features.get('std_active', 0))
    min_idle = float(features.get('min_idle', 0))
    mean_idle = float(features.get('mean_idle', 0))
    max_idle = float(features.get('max_idle', 0))
    std_idle = float(features.get('std_idle', 0))
    
    # Engineered features
    flowiat_cov = std_flowiat / (mean_flowiat + 1e-6)
    active_cov = std_active / (mean_active + 1e-6)
    idle_cov = std_idle / (mean_idle + 1e-6)
    active_idle_ratio = mean_active / (mean_idle + 1e-6)
    bytes_per_pkt = flowBytesPerSecond / (flowPktsPerSecond + 1e-6)
    fiat_range = max_fiat - min_fiat
    biat_range = max_biat - min_biat
    flowiat_range = max_flowiat - min_flowiat
    active_range = max_active - min_active
    idle_range = max_idle - min_idle
    fiat_skew_approx = (max_fiat - mean_fiat) / (mean_fiat - min_fiat + 1e-6)
    biat_skew_approx = (max_biat - mean_biat) / (mean_biat - min_biat + 1e-6)
    flowiat_skew_approx = (max_flowiat - mean_flowiat) / (mean_flowiat - min_flowiat + 1e-6)
    active_skew_approx = (max_active - mean_active) / (mean_active - min_active + 1e-6)
    idle_skew_approx = (max_idle - mean_idle) / (mean_idle - min_idle + 1e-6)
    duration_bytes_interaction = duration * flowBytesPerSecond
    duration_pkts_interaction = duration * flowPktsPerSecond
    
    return {
        'duration': duration, 'total_fiat': total_fiat, 'total_biat': total_biat,
        'min_fiat': min_fiat, 'min_biat': min_biat, 'max_fiat': max_fiat, 'max_biat': max_biat,
        'mean_fiat': mean_fiat, 'mean_biat': mean_biat,
        'flowPktsPerSecond': flowPktsPerSecond, 'flowBytesPerSecond': flowBytesPerSecond,
        'min_flowiat': min_flowiat, 'max_flowiat': max_flowiat, 'mean_flowiat': mean_flowiat, 'std_flowiat': std_flowiat,
        'min_active': min_active, 'mean_active': mean_active, 'max_active': max_active, 'std_active': std_active,
        'min_idle': min_idle, 'mean_idle': mean_idle, 'max_idle': max_idle, 'std_idle': std_idle,
        'flowiat_cov': flowiat_cov, 'active_cov': active_cov, 'idle_cov': idle_cov,
        'active_idle_ratio': active_idle_ratio, 'bytes_per_pkt': bytes_per_pkt,
        'fiat_range': fiat_range, 'biat_range': biat_range, 'flowiat_range': flowiat_range,
        'active_range': active_range, 'idle_range': idle_range,
        'fiat_skew_approx': fiat_skew_approx, 'biat_skew_approx': biat_skew_approx,
        'flowiat_skew_approx': flowiat_skew_approx, 'active_skew_approx': active_skew_approx,
        'idle_skew_approx': idle_skew_approx,
        'duration_bytes_interaction': duration_bytes_interaction,
        'duration_pkts_interaction': duration_pkts_interaction
    }

@app.route('/predict', methods=['POST'])
def predict():
    """Two-stage prediction: VPN detection + Application classification"""
    try:
        data = request.get_json()
        
        # Get selected model
        model_name = data.get('model', 'LightGBM')
        if model_name not in models:
            return jsonify({'error': f'Model {model_name} not found'}), 400
        
        model = models[model_name]
        
        # Get features and engineer them
        features = data.get('features', {})
        
        # Stage 1: VPN Detection
        is_vpn, vpn_confidence = detect_vpn(features)
        
        # Stage 2: Application Classification
        feature_vector = engineer_features(features)
        
        # Convert to numpy array
        feature_vector = np.array(feature_vector).reshape(1, -1)
        
        # Apply log1p transformation (same as training)
        feature_vector = np.log1p(feature_vector)
        
        # Handle any inf/nan
        feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Apply scaler if available
        if scaler is not None:
            feature_vector = scaler.transform(feature_vector)
        
        # Predict application type
        prediction = model.predict(feature_vector)[0]
        
        # Get probabilities if available
        probabilities = {}
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(feature_vector)[0]
            for i, label in enumerate(CLASS_LABELS):
                if i < len(proba):
                    probabilities[label] = float(proba[i]) * 100
        
        # Get class label
        if isinstance(prediction, (int, np.integer)):
            app_class = CLASS_LABELS[prediction] if prediction < len(CLASS_LABELS) else str(prediction)
        else:
            app_class = str(prediction)
        
        # Combine VPN detection with application classification
        if is_vpn is not None:
            full_class = f"VPN-{app_class}" if is_vpn else app_class
        else:
            full_class = app_class
        
        # Get QoS recommendation (based on application type, not VPN status)
        qos_recommendation = get_qos_for_traffic(app_class)
        
        # Estimate performance from input features
        performance = estimate_performance(features)
        
        return jsonify({
            'success': True,
            'prediction': full_class,
            'app_type': app_class,
            'is_vpn': is_vpn,
            'vpn_confidence': vpn_confidence if is_vpn is not None else None,
            'model': model_name,
            'probabilities': probabilities,
            'class_info': CLASS_INFO.get(app_class, {}),
            'qos': qos_recommendation,
            'performance': performance
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/qos/<traffic_class>')
def get_qos(traffic_class):
    """Get QoS recommendation for a traffic class"""
    qos = get_qos_for_traffic(traffic_class.upper())
    return jsonify(qos)

@app.route('/qos/allocate', methods=['POST'])
def allocate_bandwidth():
    """Allocate bandwidth based on traffic distribution"""
    try:
        data = request.get_json()
        traffic_counts = data.get('traffic_counts', {})
        allocation = allocate_bandwidth_for_traffic(traffic_counts)
        return jsonify(allocation)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_csv', methods=['POST'])
def predict_csv():
    """Predict from uploaded CSV file"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        model_name = request.form.get('model', 'LightGBM')
        if model_name not in models:
            return jsonify({'error': f'Model {model_name} not found'}), 400
        
        model = models[model_name]
        
        # Read CSV
        df = pd.read_csv(file)
        
        # Select only the features we need (if they exist)
        available_features = [f for f in FEATURE_NAMES if f in df.columns]
        
        if len(available_features) == 0:
            return jsonify({'error': 'No valid features found in CSV'}), 400
        
        # Fill missing features with 0
        for f in FEATURE_NAMES:
            if f not in df.columns:
                df[f] = 0
        
        X = df[FEATURE_NAMES].values
        
        # Apply log1p transformation
        X = np.log1p(X)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Predict
        predictions = model.predict(X)
        
        # Convert to labels
        results = []
        for i, pred in enumerate(predictions):
            if isinstance(pred, (int, np.integer)):
                label = CLASS_LABELS[pred] if pred < len(CLASS_LABELS) else str(pred)
            else:
                label = str(pred)
            results.append({
                'row': i + 1,
                'prediction': label,
                'info': CLASS_INFO.get(label, {})
            })
        
        # Summary
        summary = {}
        for r in results:
            label = r['prediction']
            summary[label] = summary.get(label, 0) + 1
        
        return jsonify({
            'success': True,
            'total_rows': len(results),
            'results': results[:100],  # Limit to first 100 for display
            'summary': summary,
            'model': model_name
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/models')
def get_models():
    """Get list of available models"""
    return jsonify({
        'models': list(models.keys()),
        'classes': CLASS_LABELS,
        'features': FEATURE_NAMES
    })

if __name__ == '__main__':
    print("=" * 50)
    print("Mobile Network Traffic Classification")
    print("=" * 50)
    load_models()
    print("\nStarting web server...")
    print("Open http://localhost:5000 in your browser")
    print("=" * 50)
    app.run(debug=True, host='0.0.0.0', port=5000)

