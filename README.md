# 🌐 Mobile Network Traffic Classification for QoS Optimization

Machine Learning-based traffic classification system for mobile network Quality of Service (QoS) optimization.

## 📊 Project Overview

This project implements a **two-stage classification system**:
1. **VPN Detection**: Binary classification (VPN vs Non-VPN)
2. **Application Classification**: 7-class classification (BROWSING, CHAT, FILE_TRANSFER, MAIL, P2P, STREAMING, VOIP)

### 🎯 Accuracy Results

| Model | Accuracy |
|-------|----------|
| RandomForest | **96.92%** |
| LightGBM | **96.79%** |
| XGBoost | **95.73%** |
| VPN Detector | **77.46%** |

## 🚀 Features

- **Web Interface**: Beautiful Flask-based UI for real-time classification
- **Multiple Models**: LightGBM, RandomForest, XGBoost
- **QoS Recommendations**: Priority, bandwidth, latency, jitter, DSCP marking
- **CSV Upload**: Batch classification support
- **Two-Stage Classification**: VPN detection + Application type

## 📁 Project Structure

```
Mobile/
├── app.py                 # Flask web application
├── main.py                # Training pipeline
├── train_vpn_detector.py  # VPN detector training
├── src/
│   ├── config.py          # Configuration & hyperparameters
│   ├── preprocessing.py   # Data preprocessing & feature engineering
│   ├── models.py          # Model training & evaluation
│   └── qos.py             # QoS optimization module
├── templates/
│   └── index.html         # Web UI template
├── static/
│   ├── css/style.css      # Styling
│   └── js/app.js          # Frontend JavaScript
├── models/                # Trained models (not in repo)
├── data/                  # Dataset (not in repo)
└── plots/                 # Visualization outputs
```

## 🛠️ Installation

1. **Clone the repository**
```bash
git clone https://github.com/MehmetEmirAlbayrak/mobile-traffic-classification.git
cd mobile-traffic-classification
```

2. **Create virtual environment**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download Dataset**
- Download ISCX VPN-nonVPN dataset
- Place ARFF files in `data/` directory

5. **Train Models**
```bash
python main.py
python train_vpn_detector.py
```

6. **Run Web Application**
```bash
python app.py
```

Open http://localhost:5000 in your browser.

## 📊 Dataset

This project uses the **ISCX VPN-nonVPN Traffic Dataset** with flow-level features:
- Duration, packet rates, byte rates
- Inter-arrival times (forward/backward)
- Active/idle time statistics

## 🔧 Technologies

- **Python 3.8+**
- **Scikit-learn** - ML framework
- **LightGBM** - Gradient boosting
- **XGBoost** - Gradient boosting
- **Flask** - Web framework
- **Pandas/NumPy** - Data processing

## 👥 Team

- Helin Saygılı (240104004980)
- Mehmet Emir Albayrak (210104004033)
- Ahmet Sadri Güler (200104004015)

## 📚 Course

**CSE476 - Mobile Communication Networks**  
Instructor: Hasari ÇELEBİ

## 📄 License

This project is for educational purposes.
