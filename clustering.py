import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score


class SoilClusteringEngine:
    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = None
        self.kmeans = None
        self.scaled_data = None
        self.pca_data = None
        self.cluster_labels = None
        self.feature_names = None
        
    def preprocess_data(self, df, feature_columns):
        self.feature_names = feature_columns
        data = df[feature_columns].values
        self.scaled_data = self.scaler.fit_transform(data)
        return self.scaled_data
    
    def apply_pca(self, n_components=3):
        if self.scaled_data is None:
            raise ValueError("Data must be preprocessed first")
        
        self.pca = PCA(n_components=n_components)
        self.pca_data = self.pca.fit_transform(self.scaled_data)
        
        variance_explained = self.pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(variance_explained)
        
        return {
            'pca_data': self.pca_data,
            'variance_explained': variance_explained,
            'cumulative_variance': cumulative_variance,
            'components': self.pca.components_
        }
    
    def calculate_elbow_scores(self, max_clusters=10):
        if self.scaled_data is None:
            raise ValueError("Data must be preprocessed first")
        
        inertias = []
        silhouette_scores = []
        davies_bouldin_scores = []
        k_range = range(2, min(max_clusters + 1, len(self.scaled_data)))
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.scaled_data)
            
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(self.scaled_data, labels))
            davies_bouldin_scores.append(davies_bouldin_score(self.scaled_data, labels))
        
        return {
            'k_values': list(k_range),
            'inertias': inertias,
            'silhouette_scores': silhouette_scores,
            'davies_bouldin_scores': davies_bouldin_scores
        }
    
    def perform_clustering(self, n_clusters):
        if self.scaled_data is None:
            raise ValueError("Data must be preprocessed first")
        
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.cluster_labels = self.kmeans.fit_predict(self.scaled_data)
        
        silhouette = silhouette_score(self.scaled_data, self.cluster_labels)
        davies_bouldin = davies_bouldin_score(self.scaled_data, self.cluster_labels)
        
        return {
            'labels': self.cluster_labels,
            'centroids': self.kmeans.cluster_centers_,
            'inertia': self.kmeans.inertia_,
            'silhouette_score': silhouette,
            'davies_bouldin_score': davies_bouldin
        }
    
    def get_cluster_statistics(self, df, feature_columns):
        if self.cluster_labels is None:
            raise ValueError("Clustering must be performed first")
        
        df_with_clusters = df.copy()
        df_with_clusters['Cluster'] = self.cluster_labels
        
        cluster_stats = df_with_clusters.groupby('Cluster')[feature_columns].agg([
            'mean', 'std', 'min', 'max', 'count'
        ]).round(3)
        
        cluster_sizes = df_with_clusters['Cluster'].value_counts().sort_index()
        
        return {
            'statistics': cluster_stats,
            'sizes': cluster_sizes,
            'df_with_clusters': df_with_clusters
        }
    
    def get_feature_importance(self):
        if self.pca is None or self.feature_names is None:
            raise ValueError("PCA must be applied first")
        
        components_df = pd.DataFrame(
            self.pca.components_.T,
            columns=[f'PC{i+1}' for i in range(self.pca.n_components_)],
            index=self.feature_names
        )
        
        feature_importance = np.abs(self.pca.components_).sum(axis=0)
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': feature_importance / feature_importance.sum()
        }).sort_values('Importance', ascending=False)
        
        return {
            'components': components_df,
            'importance': importance_df
        }
