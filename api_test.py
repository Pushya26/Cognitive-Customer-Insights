# Copyright (c) 2025 [B. C. PUSHYA]
# Licensed under the MIT License (see LICENSE for details)

import requests
import json
import numpy as np

def test_api():
    try:
        # Get expected feature count from the API
        info_response = requests.get("http://localhost:5000/")
        print("Server info:", info_response.text)

        # Create properly formatted test data
        test_data = {
            "numerical_features": np.random.rand(7).tolist(),  # Example with 7 features
            "text": "Excellent customer service"  # Optional
        }

        print("\nSending test data:", json.dumps(test_data, indent=2))
        
        response = requests.post(
            "http://localhost:5000/predict",
            json=test_data,
            timeout=10
        )

        print(f"\nStatus Code: {response.status_code}")
        
        if response.status_code == 200:
            print("Successful Response:")
            print(json.dumps(response.json(), indent=2))
        else:
            print("Error Response:")
            print(response.text)

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {str(e)}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    test_api()
