"""
Test the new return_image feature
"""
import requests
import json
import base64
from pathlib import Path

API_URL = "http://localhost:8000/api/verify"

def test_with_annotated_image():
    """Test verification with annotated image"""
    print("="*60)
    print("Testing Certificate Verification with Annotated Image")
    print("="*60)
    
    # Use one of the seal images
    test_file = "cropped_seals/temp_cert_264196_seal_1.png"
    
    if not Path(test_file).exists():
        print(f"❌ Test file not found: {test_file}")
        return
    
    print(f"\n📄 Testing with: {test_file}")
    print(f"📊 Testing WITH annotated image (return_image=true)\n")
    
    try:
        with open(test_file, 'rb') as f:
            files = {'files': (Path(test_file).name, f, 'image/png')}
            # Add return_image=true parameter
            params = {'return_image': 'true'}
            response = requests.post(API_URL, files=files, params=params)
        
        print(f"✅ Status: {response.status_code}")
        result = response.json()
        
        # Display main results
        print(f"\n📊 Verification Results:")
        print(f"   Decision: {result.get('decision')}")
        print(f"   Confidence: {result.get('confidence')}")
        print(f"   Reason: {result.get('reason')}")
        print(f"   Processing Time: {result.get('processing_time_seconds')}s")
        
        # Check if annotated image is present
        if 'annotated_image' in result:
            print(f"\n🎨 Annotated Image:")
            print(f"   ✅ Base64 image included")
            print(f"   Size: {len(result['annotated_image'])} characters")
            
            # Save annotated image to file
            img_data = base64.b64decode(result['annotated_image'])
            output_path = "annotated_certificate.png"
            with open(output_path, 'wb') as f:
                f.write(img_data)
            print(f"   💾 Saved to: {output_path}")
            
            if 'annotated_image_url' in result:
                print(f"   🌐 Data URL available (for direct display in browser)")
                print(f"   URL length: {len(result['annotated_image_url'])} characters")
        else:
            print(f"\n⚠️  No annotated image in response")
        
        # Show seal detection details
        seal_info = result.get('details', {}).get('seal_verification', {})
        if seal_info:
            print(f"\n🔍 Seal Detection:")
            print(f"   Total seals: {seal_info.get('total_seals', 0)}")
            print(f"   Authentic: {seal_info.get('authentic_seals', 0)}")
            print(f"   Fake: {seal_info.get('fake_seals', 0)}")
            print(f"   Method: {seal_info.get('detection_method', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_without_annotated_image():
    """Test verification without annotated image (default behavior)"""
    print("\n" + "="*60)
    print("Testing WITHOUT Annotated Image (return_image=false)")
    print("="*60)
    
    test_file = "cropped_seals/temp_cert_264196_seal_1.png"
    
    print(f"\n📄 Testing with: {test_file}")
    print(f"📊 Testing WITHOUT annotated image (default)\n")
    
    try:
        with open(test_file, 'rb') as f:
            files = {'files': (Path(test_file).name, f, 'image/png')}
            # Don't specify return_image parameter (defaults to false)
            response = requests.post(API_URL, files=files)
        
        result = response.json()
        print(f"✅ Status: {response.status_code}")
        print(f"   Decision: {result.get('decision')}")
        print(f"   Time: {result.get('processing_time_seconds')}s")
        
        if 'annotated_image' in result:
            print(f"   ⚠️  Annotated image included (unexpected)")
        else:
            print(f"   ✅ No annotated image (as expected)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("\n🧪 ANNOTATED IMAGE API TESTING\n")
    
    # Wait for API
    import time
    print("⏳ Waiting for API to start...")
    time.sleep(5)
    
    # Check API health
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        print(f"✅ API is running\n")
    except:
        print(f"❌ API not running. Start it first!")
        exit(1)
    
    # Run tests
    test1 = test_with_annotated_image()
    test2 = test_without_annotated_image()
    
    print("\n" + "="*60)
    if test1 and test2:
        print("✅ ALL TESTS PASSED!")
        print("\n💡 Open 'annotated_certificate.png' to see the result!")
    else:
        print("❌ SOME TESTS FAILED")
    print("="*60)
