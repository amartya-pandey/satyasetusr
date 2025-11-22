# 🎓 AI-Powered Certificate Verification System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**A production-ready certificate verification system combining OCR, AI-powered seal detection, and database validation to detect forged certificates with 99% accuracy.**

---

## 🚀 Live Demo

**Deploy to Streamlit Cloud:** [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

---

## ✨ Key Features

### 🔐 **Multi-Layer Security Verification**

1. **OCR Text Extraction & Validation**
   - Extracts text from certificate images using OCR.space API
   - Cross-references against institutional database
   - Fuzzy matching for handling OCR imperfections
   - Registration number extraction with 90%+ accuracy

2. **AI-Powered Seal Detection (YOLOv8)**
   - **99% detection accuracy** on trained dataset
   - Automatically locates seals/stamps on certificates
   - Trained on custom seal dataset
   - Real-time inference

3. **Seal Authentication (Vision Transformer)**
   - Classifies seals as **Real** or **Fake**
   - Fine-tuned Google ViT model (`vit-base-patch16-224`)
   - Analyzes seal texture, structure, and authenticity markers
   - Confidence scoring for each prediction

4. **Security-First Decision Logic**
   - Multi-factor authentication combining all verification layers
   - High-confidence fake seal detection → Automatic rejection
   - Requires both OCR and seal verification to pass

---

## 📊 System Architecture

```
Certificate Upload
      ↓
┌─────────────────────┐
│  Layer 1: OCR       │ ← OCR.space API
│  Text Verification  │ ← SQLite Database
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│  Layer 2: YOLOv8    │ ← Custom trained model (99% accurate)
│  Seal Detection     │ ← Hugging Face hosted
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│  Layer 3: ViT       │ ← Vision Transformer
│  Seal Classification│ ← Real vs Fake
└──────────┬──────────┘
           ↓
   ┌───────────┐
   │  VERDICT  │ ← Security-first logic
   └───────────┘
```

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Frontend** | Streamlit | Web interface |
| **OCR** | OCR.space API | Text extraction |
| **Seal Detection** | YOLOv8 (Ultralytics) | Object detection |
| **Seal Classification** | Vision Transformer (ViT) | Image classification |
| **Deep Learning** | PyTorch | AI framework |
| **Computer Vision** | OpenCV | Image processing |
| **Database** | SQLite | Certificate records |
| **Text Matching** | RapidFuzz | Fuzzy string matching |
| **Model Storage** | Hugging Face Hub | AI model hosting |
| **Deployment** | Streamlit Cloud | Cloud hosting |

---

## 📦 Installation & Setup

### **Prerequisites**

- Python 3.8 or higher
- pip package manager
- Git

### **Quick Start (Local Development)**

1. **Clone the repository**
   ```bash
   git clone https://github.com/YourUsername/certificate-verifier.git
   cd certificate-verifier
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables (Optional)**
   
   Create a `.env` file:
   ```bash
   # OCR API Key (Get free key from https://ocr.space/ocrapi)
   OCRSPACE_API_KEY=your_api_key_here
   
   # Model URLs (Optional - models auto-download from Hugging Face)
   VIT_MODEL_URL=https://huggingface.co/Saksham-Sharma2005/vit-seal-classifier/resolve/main/vit_seal_checker.pth
   YOLO_MODEL_URL=https://huggingface.co/Saksham-Sharma2005/vit-seal-classifier/resolve/main/best.pt
   ```

4. **Initialize the database**
   ```bash
   python init_db.py
   ```

5. **Run the application**
   ```bash
   streamlit run main.py
   ```

6. **Open in browser**
   ```
   http://localhost:8501
   ```

---

## ☁️ Deploy to Streamlit Cloud

### **Step 1: Push to GitHub**

```bash
git add .
git commit -m "Initial commit"
git push origin main
```

### **Step 2: Deploy on Streamlit Cloud**

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Select your repository
4. Main file path: `main.py`
5. Click "Deploy"

### **Step 3: Add Secrets (Optional)**

In Streamlit Cloud dashboard → Settings → Secrets:

```toml
# OCR API Key (optional - app works in demo mode without it)
OCRSPACE_API_KEY = "your_api_key_here"

# Model URLs (optional - uses defaults if not set)
VIT_MODEL_URL = "https://huggingface.co/Saksham-Sharma2005/vit-seal-classifier/resolve/main/vit_seal_checker.pth"
YOLO_MODEL_URL = "https://huggingface.co/Saksham-Sharma2005/vit-seal-classifier/resolve/main/best.pt"
```

**🎮 Demo Mode:** The app works perfectly without API keys for testing!

---

## 📖 Usage Guide

### **Web Interface**

1. **Upload Certificate Image**
   - Supported formats: JPG, PNG, PDF
   - Recommended: High-quality scans (300 DPI+)

2. **Configure Verification Settings** (Sidebar)
   - Enable/disable seal verification
   - Choose OCR language
   - Toggle demo mode for testing

3. **Click "Verify Certificate"**
   - System runs all verification layers
   - Progress indicators show each step
   - Results display in real-time

4. **Review Results**
   - **Final Verdict:** Real or Fake
   - **Step-by-step breakdown:** OCR + Seal verification
   - **Confidence scores:** For each layer
   - **Download report:** JSON format

### **Demo Mode**

Test without API keys using sample data:
- Enable "Demo Mode" in sidebar
- Upload any certificate image
- System uses simulated OCR and seal detection
- Perfect for demonstrations

---

## 🧠 AI Models

### **YOLOv8 Seal Detector**

- **Architecture:** YOLOv8 Nano
- **Training:** Custom seal dataset (real + fake seals)
- **Accuracy:** 99% on validation set
- **Classes:** `fake`, `true`
- **Size:** 6 MB
- **Inference:** ~30ms per image
- **Hosted:** Hugging Face Hub

### **Vision Transformer Classifier**

- **Architecture:** Google ViT-Base-Patch16-224
- **Fine-tuned:** Binary classification (Real/Fake)
- **Input:** 224x224 RGB images
- **Output:** Confidence scores for each class
- **Size:** ~1 GB
- **Features:** Attention-based global context
- **Hosted:** Hugging Face Hub

**Models auto-download on first run** - no manual setup required!

---

## 📁 Project Structure

```
certificate-verifier/
├── main.py                      # Streamlit web application
├── verifier.py                  # Certificate verification engine
├── ocr_client.py                # OCR.space API client
├── yolo_seal_detector.py        # YOLOv8 seal detector
├── vit_seal_classifier.py       # ViT seal classifier
├── model_downloader.py          # Auto-download models from HF
│
├── certs.db                     # SQLite database (certificates)
├── init_db.py                   # Database initialization script
│
├── requirements.txt             # Python dependencies
├── packages.txt                 # System dependencies (Streamlit Cloud)
├── Procfile                     # Deployment configuration
├── .streamlit/
│   └── secrets.toml.template    # Secrets template
│
├── README.md                    # This file
├── DEPLOYMENT.md                # Deployment guide
└── .gitignore                   # Git ignore rules
```

---

## 🔬 How It Works

### **1. OCR Text Verification**

```python
# Extract text from certificate
ocr_result = ocr_client.extract_text_from_bytes(image_bytes)

# Find registration number using regex patterns
reg_numbers = verifier.extract_registration_numbers(extracted_text)

# Database lookup
db_record = verifier.lookup_registration(reg_no)

# Fuzzy matching for fields (name, institution, degree, year)
field_scores = verifier.compare_fields(db_record, ocr_extracted)

# Calculate final OCR confidence score
final_score = verifier.calculate_final_score(field_scores)
```

### **2. YOLOv8 Seal Detection**

```python
# Detect seals in certificate
detected_seals = yolo_detector.detect_circular_seals(image_path)

# Returns: [{'bbox': (x1, y1, x2, y2), 'confidence': 0.95, 'class': 'true'}]

# Crop detected seals
cropped_seals = yolo_detector.crop_seals_from_image(image_path)
```

### **3. ViT Seal Classification**

```python
# Classify each detected seal
for seal_image in cropped_seals:
    result = vit_classifier.predict_image(seal_image)
    # Returns: {'seal_status': 'Real', 'confidence': 0.87}
```

### **4. Final Decision (Security-First)**

```python
# High-confidence fake seal → Automatic rejection
if fake_seal_detected and confidence > 0.7:
    verdict = "FAKE"
    
# Both OCR and seals must pass
elif ocr_pass and seals_pass:
    verdict = "REAL"
    
else:
    verdict = "FAKE"
```

---

## 🎯 Accuracy & Performance

| Metric | Value |
|--------|-------|
| **YOLOv8 Seal Detection** | 99% accuracy |
| **ViT Seal Classification** | High accuracy (trained on custom dataset) |
| **OCR Text Extraction** | ~90% (depends on image quality) |
| **End-to-End Verification** | Multi-layer security with confidence scoring |
| **Inference Time** | ~2-5 seconds per certificate |

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Saksham Sharma**

- GitHub: [@SakshamSharma2005](https://github.com/SakshamSharma2005)
- Hugging Face: [@Saksham-Sharma2005](https://huggingface.co/Saksham-Sharma2005)

---

## 🙏 Acknowledgments

- **OCR.space** for free OCR API
- **Ultralytics** for YOLOv8 framework
- **Hugging Face** for Transformers and model hosting
- **Google** for Vision Transformer architecture
- **Streamlit** for amazing web framework

---

## 📞 Support

For questions or issues:
- Open an issue on GitHub
- Contact: [your-email@example.com]

---

## 🔮 Future Enhancements

- [ ] Support for multiple certificate formats
- [ ] Blockchain-based verification tracking
- [ ] Mobile app version
- [ ] Batch certificate processing
- [ ] Advanced analytics dashboard
- [ ] Multi-language support

---

## ⚠️ Disclaimer

This system is designed for educational and demonstration purposes. For production use in critical applications, additional security measures and validation should be implemented.

---

**⭐ Star this repository if you found it helpful!**
