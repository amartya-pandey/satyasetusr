# 🚀 Deployment Guide - Streamlit Cloud

This guide will help you deploy the Certificate Verification System to Streamlit Cloud in minutes.

---

## 📋 Prerequisites

- GitHub account
- Streamlit Cloud account (sign up at [share.streamlit.io](https://share.streamlit.io))
- This repository pushed to your GitHub

---

## 🎯 Quick Deploy (5 Minutes)

### **Step 1: Prepare Your Repository**

1. **Clone/Fork this repository**
   ```bash
   git clone https://github.com/YourUsername/certificate-verifier.git
   cd certificate-verifier
   ```

2. **Verify essential files exist**
   ```
   ✅ main.py
   ✅ requirements.txt
   ✅ packages.txt (for system dependencies)
   ✅ .streamlit/ directory (optional)
   ```

3. **Push to your GitHub**
   ```bash
   git add .
   git commit -m "Ready for deployment"
   git push origin main
   ```

---

### **Step 2: Deploy on Streamlit Cloud**

1. **Go to Streamlit Cloud**
   - Visit: [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub

2. **Create New App**
   - Click **"New app"** button
   - Repository: Select `certificate-verifier`
   - Branch: `main`
   - Main file path: `main.py`
   - App URL: Choose custom URL (optional)

3. **Click "Deploy!"**
   - Streamlit will install dependencies
   - Build process takes 2-3 minutes
   - Models will auto-download from Hugging Face on first run

---

### **Step 3: Configure Secrets (Optional)**

Add secrets for OCR API (optional - app works in demo mode without it):

1. **In Streamlit Cloud Dashboard:**
   - Go to your app
   - Click ⚙️ Settings → Secrets

2. **Add the following:**
   ```toml
   # OCR API Key (get free key from https://ocr.space/ocrapi)
   OCRSPACE_API_KEY = "your_api_key_here"
   
   # Model URLs (optional - uses defaults if not specified)
   VIT_MODEL_URL = "https://huggingface.co/Saksham-Sharma2005/vit-seal-classifier/resolve/main/vit_seal_checker.pth"
   YOLO_MODEL_URL = "https://huggingface.co/Saksham-Sharma2005/vit-seal-classifier/resolve/main/best.pt"
   ```

3. **Save** - App will automatically redeploy

---

## 🎮 Demo Mode

**The app works perfectly without any API keys!**

- ✅ YOLOv8 seal detection (uses Ultralytics default or downloads custom model)
- ✅ Sample OCR data for testing
- ✅ All UI features functional
- ✅ Perfect for demonstrations and testing

---

## 🔧 Configuration Files Explained

### **requirements.txt**
```txt
streamlit>=1.28.0
requests>=2.31.0
pillow>=10.0.0
opencv-python-headless>=4.8.0
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
ultralytics>=8.0.0
rapidfuzz>=3.0.0
python-dotenv>=1.0.0
```

### **packages.txt** (System dependencies for Streamlit Cloud)
```txt
libgl1-mesa-glx
libglib2.0-0
```

### **Procfile** (Optional)
```
web: streamlit run main.py --server.port=$PORT --server.address=0.0.0.0
```

---

## 📊 Model Auto-Download

Models are hosted on Hugging Face and download automatically:

### **On First Run:**

1. **ViT Model (~1 GB)**
   - Downloads from: `https://huggingface.co/Saksham-Sharma2005/vit-seal-classifier/resolve/main/vit_seal_checker.pth`
   - Cached for future runs
   - One-time download

2. **YOLOv8 Model (~6 MB)**
   - Downloads from: `https://huggingface.co/Saksham-Sharma2005/vit-seal-classifier/resolve/main/best.pt`
   - Quick download
   - Cached locally

**Total first-run time:** ~5-10 minutes (depending on bandwidth)

**Subsequent runs:** Instant (models are cached)

---

## 🚨 Troubleshooting

### **Issue: App crashes during model download**
**Solution:** Streamlit Cloud has memory limits. Models download on first run and are cached. If it times out, just restart the app - cached downloads persist.

### **Issue: OCR API errors**
**Solution:** 
- Enable "Demo Mode" in sidebar for testing without API
- Get free API key from [OCR.space](https://ocr.space/ocrapi)
- Add to Streamlit secrets

### **Issue: Import errors**
**Solution:**
- Check `requirements.txt` has all dependencies
- Ensure `packages.txt` includes system libraries
- Verify Python version compatibility (3.8+)

### **Issue: Database not found**
**Solution:**
- Ensure `certs.db` is pushed to GitHub
- Or run `python init_db.py` locally first
- Database is included in repository

### **Issue: Models not downloading**
**Solution:**
- Check Hugging Face URLs are correct
- Verify internet connectivity in Streamlit Cloud
- Check logs for specific download errors

---

## 📈 Monitoring & Logs

### **View App Logs:**
1. Go to Streamlit Cloud dashboard
2. Click on your app
3. Click "Manage app" → "Logs"
4. View real-time logs

### **Check Resource Usage:**
- CPU usage
- Memory usage
- Active users
- Response times

---

## 🔄 Updating Your App

### **Automatic Deployment:**
Every push to `main` branch triggers automatic redeployment.

```bash
git add .
git commit -m "Update feature"
git push origin main
```

Streamlit Cloud detects changes and redeploys automatically.

### **Manual Reboot:**
In Streamlit Cloud dashboard → Click "Reboot app"

---

## 🌐 Custom Domain (Optional)

### **Free Streamlit Domain:**
- Your app gets: `your-app-name.streamlit.app`
- Customize in deployment settings

### **Custom Domain:**
- Available on Streamlit Cloud paid plans
- Configure CNAME record
- Follow Streamlit's documentation

---

## 💡 Best Practices

### **1. Security**
- ✅ Never commit API keys to GitHub
- ✅ Use Streamlit secrets for sensitive data
- ✅ Add `.env` to `.gitignore`

### **2. Performance**
- ✅ Models are cached after first download
- ✅ Use `@st.cache_data` for expensive operations
- ✅ Optimize image uploads (resize before processing)

### **3. User Experience**
- ✅ Enable demo mode by default
- ✅ Show clear error messages
- ✅ Add progress indicators
- ✅ Provide usage instructions

---

## 📞 Support

### **Streamlit Cloud Support:**
- Documentation: https://docs.streamlit.io/streamlit-community-cloud
- Community Forum: https://discuss.streamlit.io
- Status Page: https://streamlitstatus.com

### **Project Support:**
- Open an issue on GitHub
- Check existing issues
- Contact maintainer

---

## 🎉 You're Live!

Once deployed, share your app:

```
🔗 Your App URL: https://your-app-name.streamlit.app
```

**Share with:**
- QR code (generated by Streamlit)
- Direct link
- Embed in documentation
- Social media

---

## 📦 Alternative Deployment Options

### **Heroku**
- More control
- Custom buildpacks
- Paid plans for more resources

### **AWS/GCP/Azure**
- Full infrastructure control
- Scalable
- Requires more setup

### **Docker**
- Containerized deployment
- Portable across platforms
- Good for on-premise deployment

---

**🚀 Congratulations! Your Certificate Verification System is now live!**

For questions or issues, please open an issue on GitHub.
