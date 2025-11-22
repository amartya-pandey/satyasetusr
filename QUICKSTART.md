# 🚀 Quick Start Guide

Get the Certificate Verification System running in **5 minutes**!

---

## ⚡ Fastest Way to Run

### **Option 1: Demo Mode (No Setup Required)**

```bash
# Clone repository
git clone https://github.com/YourUsername/certificate-verifier.git
cd certificate-verifier

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run main.py
```

**That's it!** App runs in demo mode - no API keys needed.

---

## 🔧 Full Setup (With OCR API)

### **1. Install Dependencies**

```bash
pip install -r requirements.txt
```

### **2. Get Free OCR API Key**

1. Visit: https://ocr.space/ocrapi
2. Sign up (free)
3. Copy your API key

### **3. Configure Environment**

Create `.env` file:

```bash
OCRSPACE_API_KEY=your_api_key_here
```

### **4. Initialize Database** (Optional)

```bash
python init_db.py
```

### **5. Run Application**

```bash
streamlit run main.py
```

### **6. Open Browser**

```
http://localhost:8501
```

---

## 📱 Usage

### **Basic Workflow:**

1. **Upload** certificate image (JPG/PNG)
2. **Click** "Verify Certificate"
3. **View** results with confidence scores
4. **Download** verification report (JSON)

### **Demo Mode Testing:**

1. Enable "Demo Mode" in sidebar
2. Upload any certificate image
3. System uses sample OCR data
4. See how verification works!

---

## 🎯 What Happens on First Run?

### **Automatic Downloads:**

1. **YOLOv8 Model** (~6 MB)
   - Downloads from Hugging Face
   - Takes ~10 seconds
   - Cached for future runs

2. **ViT Model** (~1 GB) (Only if seal verification enabled)
   - Downloads from Hugging Face
   - Takes ~5 minutes (depending on bandwidth)
   - Cached for future runs

**After first run:** Everything loads instantly from cache!

---

## ✅ Verification Steps

The system performs **3-layer verification**:

### **Layer 1: OCR Text Verification**
- Extracts text from certificate
- Finds registration number
- Matches against database
- Calculates confidence score

### **Layer 2: YOLOv8 Seal Detection**
- Detects seals/stamps in image
- 99% detection accuracy
- Returns bounding boxes

### **Layer 3: ViT Seal Classification**
- Classifies each seal as Real/Fake
- Uses Vision Transformer AI
- Provides confidence scores

### **Final Decision:**
- Combines all layers
- Security-first logic
- High-confidence fake → Rejection

---

## 🔍 Test Certificates

The database includes sample certificates you can test:

**Sample Registration Numbers:**
- `ABC2023001` - Saksham Sharma, DevLabs Institute
- `ABC2022007` - Prisha Verma, Global Tech University
- `1BG19CS100` - Vikram Verma, VTU (from demo mode)

---

## 🎮 Features to Try

### **In Sidebar:**

✅ **Demo Mode** - Test without API keys
✅ **Seal Verification** - Enable AI seal detection
✅ **OCR Language** - Select certificate language
✅ **System Status** - Check all components

### **After Verification:**

✅ **Detailed Results** - Step-by-step breakdown
✅ **Confidence Scores** - For each verification layer
✅ **Download Report** - JSON export
✅ **Detected Seals** - View cropped seal images

---

## 🚨 Troubleshooting

### **Issue: ModuleNotFoundError**
```bash
pip install -r requirements.txt
```

### **Issue: OCR API Error**
- Enable "Demo Mode" in sidebar, or
- Check API key in `.env` file

### **Issue: Models not downloading**
- Check internet connection
- Models download automatically on first run
- Look for download progress in terminal

### **Issue: Database error**
```bash
python init_db.py
```

---

## 📚 Next Steps

### **Deploy to Cloud:**
See [DEPLOYMENT.md](DEPLOYMENT.md) for Streamlit Cloud deployment

### **Customize:**
- Edit `certs.db` to add your certificates
- Modify verification thresholds in `verifier.py`
- Add custom regex patterns for registration numbers

### **Integrate:**
- Use as Python library
- Build REST API wrapper
- Integrate with existing systems

---

## 🤝 Need Help?

- **Documentation:** [README.md](README.md)
- **Deployment:** [DEPLOYMENT.md](DEPLOYMENT.md)
- **Issues:** Open on GitHub
- **Questions:** Contact maintainer

---

## 🎉 You're Ready!

**Start verifying certificates with AI-powered accuracy!**

```bash
streamlit run main.py
```

Happy verifying! 🎓✨
