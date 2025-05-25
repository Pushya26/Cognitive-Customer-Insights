# Copyright (c) 2025 [B. C. PUSHYA]
# Licensed under the MIT License (see LICENSE for details)

from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import joblib
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Initialize NLTK
nltk.download('wordnet')
nltk.download('stopwords')

app = Flask(__name__)

# Load models
autoencoder = tf.keras.models.load_model("autoencoder_model.h5")
kmeans = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")
try:
    tfidf = joblib.load("tfidf_vectorizer.pkl")
except:
    tfidf = None

def preprocess_text(text):
    """Clean and preprocess text input"""
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Validate input
        if not data or 'numerical_features' not in data:
            return jsonify({"error": "Missing numerical_features"}), 400
            
        numerical_features = np.array(data['numerical_features'])
        
        # Reshape and validate feature count
        if numerical_features.ndim == 1:
            numerical_features = numerical_features.reshape(1, -1)
            
        if numerical_features.shape[1] != scaler.n_features_in_:
            return jsonify({
                "error": f"Expected {scaler.n_features_in_} features, got {numerical_features.shape[1]}. Required features: {scaler.feature_names_in_ if hasattr(scaler, 'feature_names_in_') else 'Unknown'}"
            }), 400
        
        # Process numerical features
        scaled_features = scaler.transform(numerical_features)
        pca_features = pca.transform(scaled_features)
        
        # Process text if available
        if tfidf and 'text' in data:
            cleaned_text = preprocess_text(data['text'])
            tfidf_features = tfidf.transform([cleaned_text]).toarray()
            all_features = np.concatenate([pca_features, tfidf_features], axis=1)
        else:
            all_features = pca_features
        
        # Extract latent features
        latent_features = autoencoder.predict(all_features)
        
        # Predict cluster
        cluster = kmeans.predict(latent_features)
        
        return jsonify({
            "predicted_segment": int(cluster[0]),  # Must include this field
            "latent_features": latent_features[0].tolist()
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "prediction_failed"
        }), 500

if __name__ == '__main__':
    from waitress import serve
    print(f"Scaler expects {scaler.n_features_in_} features")
    if hasattr(scaler, 'feature_names_in_'):
        print("Feature names:", scaler.feature_names_in_)
    serve(app, host="0.0.0.0", port=5000)
