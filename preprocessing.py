# Copyright (c) 2025 [B. C. PUSHYA]
# Licensed under the MIT License (see LICENSE for details)

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_data(data_path):
    """Preprocess customer data for the segmentation model"""
    # Load data
    df = pd.read_csv(data_path)
    
    # 1. Handle missing values
    # Numerical features
    num_imputer = KNNImputer(n_neighbors=5)
    numerical_cols = df.select_dtypes(include=np.number).columns
    df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])
    
    # Categorical features
    cat_imputer = SimpleImputer(strategy='most_frequent')
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])
    
    # 2. Handle outliers using IQR method
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = np.clip(df[col], lower_bound, upper_bound)
    
    # 3. Text preprocessing
    tfidf = None
    if 'feedback' in df.columns:
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))
        
        def clean_text(text):
            text = str(text).lower()
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            words = text.split()
            words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
            return ' '.join(words)
        
        df['cleaned_text'] = df['feedback'].apply(clean_text)
        
        # TF-IDF vectorization
        tfidf = TfidfVectorizer(max_features=100)
        tfidf_features = tfidf.fit_transform(df['cleaned_text'])
        tfidf_df = pd.DataFrame(tfidf_features.toarray(), 
                              columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])])
        df = pd.concat([df, tfidf_df], axis=1)
    
    # 4. Feature scaling
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    # 5. Dimensionality reduction
    pca = PCA(n_components=0.95)  # Retain 95% variance
    numerical_pca = pca.fit_transform(df[numerical_cols])
    pca_cols = [f'pca_{i}' for i in range(numerical_pca.shape[1])]
    numerical_pca_df = pd.DataFrame(numerical_pca, columns=pca_cols)
    
    # Combine all features
    final_df = pd.concat([numerical_pca_df, df.drop(numerical_cols, axis=1)], axis=1)
    
    return final_df, scaler, pca, tfidf
