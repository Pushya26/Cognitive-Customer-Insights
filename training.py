import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import KFold
import joblib
import matplotlib.pyplot as plt
from preprocessing import preprocess_data

def train_autoencoder(data, epochs=50, batch_size=256):
    input_dim = data.shape[1]
    
    # Autoencoder architecture
    autoencoder = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(input_dim,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),  # Latent space
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(input_dim, activation='sigmoid')
    ])
    
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train with validation split
    history = autoencoder.fit(
        data, data,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=1
    )
    
    return autoencoder, history

def find_optimal_clusters(latent_features, max_k=10):
    """Returns optimal k and all silhouette scores"""
    silhouette_scores = []
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(latent_features)
        score = silhouette_score(latent_features, kmeans.labels_)
        silhouette_scores.append(score)
        print(f"Testing k={k}: Silhouette Score = {score:.4f}")  # Progress tracking
    
    optimal_k = np.argmax(silhouette_scores) + 2  # +2 because range starts at 2
    return optimal_k, silhouette_scores

def plot_silhouette_scores(silhouette_scores):
    """Visualize silhouette scores for different cluster counts"""
    plt.figure(figsize=(8,4))
    plt.plot(range(2, len(silhouette_scores)+2), silhouette_scores, 'bo-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Scores for Different Cluster Counts')
    plt.grid(True)
    plt.savefig('cluster_quality.png')
    plt.close()  # Close the figure to prevent display if running in script

def train_kmeans(latent_features, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(latent_features)
    return kmeans

def cross_validate(latent_features, n_clusters=5, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    silhouette_scores = []
    
    for train_index, test_index in kf.split(latent_features):
        X_train, X_test = latent_features[train_index], latent_features[test_index]
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(X_train)
        
        score = silhouette_score(X_test, kmeans.predict(X_test))
        silhouette_scores.append(score)
    
    avg_score = np.mean(silhouette_scores)
    return avg_score

def train_models(data_path):
    # Preprocess data
    processed_df, scaler, pca, tfidf = preprocess_data(data_path)
    numerical_data = processed_df.select_dtypes(include=np.number).values
    
    # Train autoencoder
    autoencoder, history = train_autoencoder(numerical_data)
    
    # Extract latent features
    latent_features = autoencoder.predict(numerical_data)
    
    # Find optimal clusters
    optimal_k, silhouette_scores = find_optimal_clusters(latent_features)
    print(f"\n=== Optimal Clustering Results ===")
    print(f"Optimal number of clusters: {optimal_k}")
    print(f"Silhouette Score: {silhouette_scores[optimal_k-2]:.4f}")
    
    # Visualize cluster quality
    plot_silhouette_scores(silhouette_scores)
    
    # Save metrics to file
    with open("clustering_metrics.txt", "w") as f:
        f.write(f"Optimal Clusters: {optimal_k}\n")
        f.write(f"Silhouette Score: {silhouette_scores[optimal_k-2]:.4f}\n")
    
    # Train final KMeans model
    kmeans = train_kmeans(latent_features, optimal_k)
    
    # Cross-validation
    cv_score = cross_validate(latent_features, optimal_k)
    print(f"Average Silhouette Score from Cross-Validation: {cv_score:.4f}")
    
    # Save models
    autoencoder.save("autoencoder_model.h5")
    joblib.dump(kmeans, "kmeans_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    if tfidf:
        joblib.dump(tfidf, "tfidf_vectorizer.pkl")
    joblib.dump(pca, "pca.pkl")
    
    return autoencoder, kmeans, scaler, tfidf, pca

if __name__ == '__main__':
    data_path = "data/customer_data.csv"  # Update with your path
    autoencoder, kmeans, scaler, tfidf, pca = train_models(data_path)