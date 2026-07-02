import requests
import json
import os

CATALOG_URL = "https://tcp-us-prod-rnd.shl.com/voiceRater/shl-ai-hiring/shl_product_catalog.json"
OUTPUT_JSON_PATH = os.path.join("data", "shl_product_catalog.json")

def download_catalog():
    print(f"Downloading official SHL product catalog from {CATALOG_URL}...")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }
    
    try:
        # Silence the SSL warning for a cleaner terminal
        requests.packages.urllib3.disable_warnings()
        
        response = requests.get(CATALOG_URL, headers=headers, verify=False)
        
        if response.status_code == 200:
            # strict=False tells Python to ignore invalid control characters in the JSON string
            catalog_data = json.loads(response.text, strict=False)
            
            # Save the clean JSON
            with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
                json.dump(catalog_data, f, indent=4)
                
            print(f"Successfully downloaded catalog! Total records: {len(catalog_data)}")
            print(f"Saved to {OUTPUT_JSON_PATH}")
            
            # Print a sample record to see the keys
            if isinstance(catalog_data, list) and len(catalog_data) > 0:
                print("\nSample Product Data Structure Keys:")
                print(list(catalog_data[0].keys()))
                
        else:
            print(f"Failed to download. Status code: {response.status_code}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    download_catalog()