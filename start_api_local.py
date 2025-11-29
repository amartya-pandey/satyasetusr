"""
Local FastAPI Server Startup Script
Runs the certificate verification API with .pth model
"""

import os
import sys
import subprocess
from pathlib import Path

def check_requirements():
    """Check if required files exist"""
    print("🔍 Checking requirements...")
    
    required_files = [
        "api.py",
        "vit_seal_checker.pth",
        "vit_seal_classifier.py",
        "yolo_seal_detector.py",
        "yolo_seal_model/best.pt"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
            print(f"   ❌ Missing: {file}")
        else:
            print(f"   ✅ Found: {file}")
    
    if missing_files:
        print(f"\n⚠️  Warning: {len(missing_files)} required file(s) missing")
        print("Some features may not work correctly.")
    else:
        print("\n✅ All required files present!")
    
    return len(missing_files) == 0

def check_dependencies():
    """Check if required Python packages are installed"""
    print("\n🔍 Checking Python dependencies...")
    
    required_packages = [
        "fastapi",
        "uvicorn",
        "torch",
        "torchvision",
        "transformers",
        "pillow",
        "ultralytics"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All dependencies installed!")
        return True

def start_server(port=8000):
    """Start the FastAPI server"""
    print(f"\n🚀 Starting FastAPI server on port {port}...")
    print("="*60)
    print(f"📦 Model: vit_seal_checker.pth (PyTorch)")
    print(f"🌐 API URL: http://localhost:{port}")
    print(f"📚 Docs: http://localhost:{port}/api/docs")
    print(f"🔍 Health: http://localhost:{port}/api/health")
    print("="*60)
    print("\nPress Ctrl+C to stop the server\n")
    
    try:
        # Run the API server
        subprocess.run([
            sys.executable,
            "api.py"
        ], env={**os.environ, "PORT": str(port)})
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")

def main():
    """Main function"""
    print("\n" + "="*60)
    print("🎯 Certificate Verification API - Local Server")
    print("="*60)
    
    # Check requirements
    files_ok = check_requirements()
    deps_ok = check_dependencies()
    
    if not deps_ok:
        print("\n❌ Cannot start server: Missing dependencies")
        print("Run: pip install -r requirements.txt")
        sys.exit(1)
    
    if not files_ok:
        response = input("\n⚠️  Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Exiting...")
            sys.exit(1)
    
    # Get port from command line or use default
    port = 8000
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"⚠️  Invalid port: {sys.argv[1]}, using default 8000")
    
    # Start server
    start_server(port)

if __name__ == "__main__":
    main()
