"""
Simple API Testing Script
Test the certificate verification API locally
"""

import requests
import sys
from pathlib import Path

API_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("\n🔍 Testing Health Endpoint...")
    try:
        response = requests.get(f"{API_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_root():
    """Test root endpoint"""
    print("\n🔍 Testing Root Endpoint...")
    try:
        response = requests.get(API_URL)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_api_status():
    """Test API status endpoint"""
    print("\n🔍 Testing API Status Endpoint...")
    try:
        response = requests.get(f"{API_URL}/api/status")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_verify(image_path):
    """Test certificate verification"""
    print(f"\n🔍 Testing Verification with: {image_path}")
    
    if not Path(image_path).exists():
        print(f"❌ File not found: {image_path}")
        return False
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': (Path(image_path).name, f, 'image/jpeg')}
            response = requests.post(
                f"{API_URL}/api/verify",
                files=files,
                data={'enable_seal_verification': 'true'}
            )
        
        print(f"Status: {response.status_code}")
        result = response.json()
        
        if result.get('success'):
            print(f"\n✅ Decision: {result['decision']}")
            print(f"📊 Confidence: {result['confidence']}")
            print(f"💬 Reason: {result['reason']}")
            
            if 'details' in result:
                details = result['details']
                print(f"\n📋 Details:")
                print(f"  - Registration: {details.get('registration_number', 'N/A')}")
                print(f"  - Database Match: {details.get('database_match', False)}")
                
                if 'seal_verification' in details and details['seal_verification']:
                    seal = details['seal_verification']
                    print(f"  - Seal Status: {seal.get('seal_status', 'N/A')}")
                    print(f"  - Total Seals: {seal.get('total_seals', 0)}")
        else:
            print(f"❌ Error: {result.get('error', 'Unknown error')}")
            print(f"💬 Message: {result.get('message', 'No message')}")
        
        return response.status_code == 200
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Main test runner"""
    print("="*60)
    print("🎯 Certificate Verification API - Test Suite")
    print("="*60)
    
    # Test endpoints
    results = {
        "Root": test_root(),
        "Health": test_health(),
        "Status": test_api_status()
    }
    
    # Test verification if image provided
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        results["Verify"] = test_verify(image_path)
    else:
        print("\n💡 Tip: Run with image path to test verification:")
        print("   python test_api.py path/to/certificate.jpg")
    
    # Summary
    print("\n" + "="*60)
    print("📊 Test Results:")
    print("="*60)
    for test, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test:20s} {status}")
    print("="*60)
    
    all_passed = all(results.values())
    print(f"\n{'🎉 All tests passed!' if all_passed else '⚠️ Some tests failed'}")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
