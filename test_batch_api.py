"""
Test script for batch certificate verification API
Tests both single and multiple certificate uploads
"""

import requests
import os
from pathlib import Path

# API endpoint
API_URL = "http://localhost:8000/api/verify"

def test_single_certificate():
    """Test single certificate upload"""
    print("\n" + "="*60)
    print("TEST 1: Single Certificate Upload")
    print("="*60)
    
    # Find a test certificate image
    test_files = list(Path(".").glob("*.jpg")) + list(Path(".").glob("*.png"))
    
    if not test_files:
        print("❌ No test images found. Add a .jpg or .png certificate image.")
        return
    
    test_file = test_files[0]
    print(f"📄 Testing with: {test_file.name}")
    
    try:
        with open(test_file, 'rb') as f:
            files = {'files': (test_file.name, f, 'image/jpeg')}
            response = requests.post(API_URL, files=files)
        
        print(f"✅ Status: {response.status_code}")
        result = response.json()
        
        print(f"\n📊 Result:")
        print(f"   Decision: {result.get('decision')}")
        print(f"   Confidence: {result.get('confidence')}")
        print(f"   Reason: {result.get('reason')}")
        print(f"   Processing Time: {result.get('processing_time_seconds')}s")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def test_batch_certificates():
    """Test batch certificate upload"""
    print("\n" + "="*60)
    print("TEST 2: Batch Certificate Upload")
    print("="*60)
    
    # Find multiple test images
    test_files = list(Path(".").glob("*.jpg"))[:5] + list(Path(".").glob("*.png"))[:5]
    
    if len(test_files) < 2:
        print("❌ Need at least 2 test images for batch test.")
        return
    
    # Limit to 5 files for testing
    test_files = test_files[:5]
    print(f"📄 Testing with {len(test_files)} certificates:")
    for f in test_files:
        print(f"   - {f.name}")
    
    try:
        # Prepare multiple files
        files = []
        file_handles = []
        
        for test_file in test_files:
            fh = open(test_file, 'rb')
            file_handles.append(fh)
            files.append(('files', (test_file.name, fh, 'image/jpeg')))
        
        response = requests.post(API_URL, files=files)
        
        # Close file handles
        for fh in file_handles:
            fh.close()
        
        print(f"✅ Status: {response.status_code}")
        result = response.json()
        
        if result.get('batch'):
            print(f"\n📊 Batch Results:")
            print(f"   Total: {result['total_certificates']}")
            print(f"   Processed: {result['processed']}")
            print(f"   Failed: {result['failed']}")
            print(f"   Total Time: {result['summary']['total_processing_time_seconds']}s")
            print(f"   Average Confidence: {result['summary']['average_confidence']}")
            
            print(f"\n📈 Summary:")
            print(f"   ✅ Authentic: {result['summary']['authentic_count']}")
            print(f"   ❌ Fake: {result['summary']['fake_count']}")
            print(f"   ⚠️  Suspicious: {result['summary']['suspicious_count']}")
            print(f"   🔴 Errors: {result['summary']['error_count']}")
            
            print(f"\n📋 Individual Results:")
            for idx, res in enumerate(result['results'], 1):
                status_icon = "✅" if res['success'] else "❌"
                decision = res.get('decision', 'ERROR')
                confidence = res.get('confidence', 0)
                time_taken = res.get('processing_time_seconds', 0)
                
                print(f"   {idx}. {status_icon} {res['filename']}")
                print(f"      Decision: {decision} ({confidence}) - {time_taken}s")
                if res.get('error'):
                    print(f"      Error: {res['error']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def test_error_handling():
    """Test error cases"""
    print("\n" + "="*60)
    print("TEST 3: Error Handling")
    print("="*60)
    
    # Test 1: Too many files
    print("\n📝 Test 3a: Too many files (>10)")
    try:
        test_files = list(Path(".").glob("*.jpg"))[:11]  # Try 11 files
        if len(test_files) >= 11:
            files = []
            file_handles = []
            
            for test_file in test_files:
                fh = open(test_file, 'rb')
                file_handles.append(fh)
                files.append(('files', (test_file.name, fh, 'image/jpeg')))
            
            response = requests.post(API_URL, files=files)
            
            for fh in file_handles:
                fh.close()
            
            if response.status_code == 400:
                print(f"   ✅ Correctly rejected: {response.json().get('detail')}")
            else:
                print(f"   ⚠️  Unexpected response: {response.status_code}")
        else:
            print(f"   ⏭️  Skipped (not enough files)")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 2: Invalid file type
    print("\n📝 Test 3b: Invalid file type")
    try:
        # Try uploading this Python script as if it were an image
        with open(__file__, 'rb') as f:
            files = {'files': ('test.py', f, 'text/plain')}
            response = requests.post(API_URL, files=files)
        
        result = response.json()
        if response.status_code == 400 or not result.get('success'):
            print(f"   ✅ Correctly rejected invalid file type")
        else:
            print(f"   ⚠️  Unexpected response: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("🧪 BATCH API TESTING SUITE")
    print("="*60)
    print(f"API Endpoint: {API_URL}")
    
    # Check API is running
    try:
        response = requests.get("http://localhost:8000/health")
        print(f"✅ API is running")
    except:
        print(f"❌ API is not running. Start it with: python api.py")
        return
    
    # Run tests
    test_single_certificate()
    test_batch_certificates()
    test_error_handling()
    
    print("\n" + "="*60)
    print("✅ Testing Complete!")
    print("="*60)

if __name__ == "__main__":
    main()
