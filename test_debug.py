"""Debug batch errors"""
import requests

API_URL = "http://localhost:8000/api/verify"

files = [
    ('files', open('cropped_seals/temp_cert_264196_seal_1.png', 'rb')),
    ('files', open('cropped_seals/temp_cert_264204_seal_2.png', 'rb'))
]

response = requests.post(API_URL, files=files)

for _, fh in files:
    fh.close()

result = response.json()
print("Full Response:")
import json
print(json.dumps(result, indent=2))
