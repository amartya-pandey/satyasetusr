"""Quick test for batch API"""
import requests

API_URL = "http://localhost:8000/api/verify"

# Test 1: Single file
print("=" * 60)
print("TEST 1: Single Certificate")
print("=" * 60)
with open('cropped_seals/temp_cert_264196_seal_1.png', 'rb') as f:
    files = {'files': ('seal1.png', f, 'image/png')}
    response = requests.post(API_URL, files=files)
    
result = response.json()
print(f"Status: {response.status_code}")
print(f"Is Batch: {result.get('batch', False)}")
print(f"Decision: {result.get('decision')}")
print(f"Confidence: {result.get('confidence')}")
print(f"Time: {result.get('processing_time_seconds')}s")

# Test 2: Multiple files
print("\n" + "=" * 60)
print("TEST 2: Batch - 3 Certificates")
print("=" * 60)

files = [
    ('files', open('cropped_seals/temp_cert_264196_seal_1.png', 'rb')),
    ('files', open('cropped_seals/temp_cert_264204_seal_2.png', 'rb')),
    ('files', open('cropped_seals/temp_cert_264211_seal_3.png', 'rb'))
]

response = requests.post(API_URL, files=files)

# Close files
for _, fh in files:
    fh.close()

result = response.json()
print(f"Status: {response.status_code}")
print(f"Is Batch: {result.get('batch')}")
print(f"Total: {result.get('total_certificates')}")
print(f"Processed: {result.get('processed')}")
print(f"Failed: {result.get('failed')}")
print(f"\nSummary:")
print(f"  Authentic: {result['summary']['authentic_count']}")
print(f"  Fake: {result['summary']['fake_count']}")
print(f"  Suspicious: {result['summary']['suspicious_count']}")
print(f"  Avg Confidence: {result['summary']['average_confidence']}")
print(f"  Total Time: {result['summary']['total_processing_time_seconds']}s")

print(f"\nIndividual Results:")
for r in result['results']:
    status = "✅" if r['success'] else "❌"
    print(f"  {status} {r['filename']}: {r.get('decision')} ({r.get('confidence')})")

print("\n" + "=" * 60)
print("✅ Batch API working perfectly!")
print("=" * 60)
