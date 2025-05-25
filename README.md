# Cognitive Customer Insights with Watson AI

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.6%2B-orange)
![License](https://img.shields.io/badge/license-MIT-green)

An AI-powered customer segmentation system that leverages:
- **Autoencoders** for feature extraction
- **K-Means Clustering** for customer segmentation
- **Flask API** for real-time predictions
- **Streamlit UI** for business user interaction

## Key Features
- Processes both numerical and text feedback data
- Identifies distinct customer segments
- Provides actionable insights for marketing strategies

## Project Structure
```
cognitive-customer-insights/
├── data/                    # Sample datasets
│   └── customer_data.csv    
├── # Trained models
│   ├── autoencoder_model.h5
│   ├── kmeans_model.pkl
│   ├── scaler.pkl
│   └── pca.pkl
├── # API components
│   ├── api.py               # Flask prediction API
│   └── requirements.txt     
├── # User Interface
│   ├── app.py               # Streamlit dashboard
│   └── assets/              # Visualizations
└── # Model development
    ├── training.py          # Model training script
    ├── preprocessing.py     # Data cleaning pipeline
    └── clustering_metrics.txt
```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Pushya26/cognitive-customer-insights.git
   cd cognitive-customer-insights
   ```

2. Set up virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Train Models
```bash
python training.py --data data/customer_data.csv
```
**Outputs:**
- Trained models in `models/`
- Cluster metrics in `training/clustering_metrics.txt`

### 2. Run API Server
```bash
python api.py
```
API Endpoint: `POST http://localhost:5000/predict`

### 3. Launch Web UI
```bash
streamlit run app.py
```
Access at: `http://localhost:8501`

## Sample API Request
```python
import requests

data = {
    "numerical_features": [35, 60000, 75, 5, 2, 3, 4],
    "text": "Excellent service experience"
}

response = requests.post("http://localhost:5000/predict", json=data)
print(response.json())
```

## Output Interpretation
| Segment | Typical Characteristics |
|---------|-------------------------|
| 0       | Price-sensitive, low engagement |
| 1       | Moderate spenders, occasional buyers |
| 2       | High-value, loyal customers |

## License
MIT License - See [LICENSE](LICENSE)
