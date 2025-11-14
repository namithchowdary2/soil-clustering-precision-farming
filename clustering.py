import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


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
        
        if len(feature_importance) != len(self.feature_names):
            raise ValueError(f"Feature importance length {len(feature_importance)} does not match number of features {len(self.feature_names)}")
        
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': feature_importance / feature_importance.sum()
        }).sort_values('Importance', ascending=False)
        
        return {
            'components': components_df,
            'importance': importance_df
        }
    
    def perform_dbscan(self, eps=0.5, min_samples=5):
        if self.scaled_data is None:
            raise ValueError("Data must be preprocessed first")
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(self.scaled_data)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        metrics = {
            'labels': labels,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'algorithm': 'DBSCAN'
        }
        
        if n_clusters > 1:
            valid_mask = labels != -1
            if valid_mask.sum() > 0:
                metrics['silhouette_score'] = silhouette_score(
                    self.scaled_data[valid_mask], 
                    labels[valid_mask]
                )
                metrics['davies_bouldin_score'] = davies_bouldin_score(
                    self.scaled_data[valid_mask], 
                    labels[valid_mask]
                )
                metrics['calinski_harabasz_score'] = calinski_harabasz_score(
                    self.scaled_data[valid_mask], 
                    labels[valid_mask]
                )
            else:
                metrics['silhouette_score'] = -1
                metrics['davies_bouldin_score'] = -1
                metrics['calinski_harabasz_score'] = -1
        else:
            metrics['silhouette_score'] = -1
            metrics['davies_bouldin_score'] = -1
            metrics['calinski_harabasz_score'] = -1
        
        self.cluster_labels = labels
        return metrics
    
    def perform_hierarchical(self, n_clusters, linkage='ward', metric='euclidean'):
        if self.scaled_data is None:
            raise ValueError("Data must be preprocessed first")
        
        if linkage == 'ward':
            metric = 'euclidean'
        
        hierarchical = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage,
            metric=metric
        )
        labels = hierarchical.fit_predict(self.scaled_data)
        
        silhouette = silhouette_score(self.scaled_data, labels)
        davies_bouldin = davies_bouldin_score(self.scaled_data, labels)
        calinski = calinski_harabasz_score(self.scaled_data, labels)
        
        self.cluster_labels = labels
        
        return {
            'labels': labels,
            'n_clusters': n_clusters,
            'silhouette_score': silhouette,
            'davies_bouldin_score': davies_bouldin,
            'calinski_harabasz_score': calinski,
            'algorithm': 'Hierarchical'
        }
    
    def perform_gmm(self, n_components, covariance_type='full', random_state=42):
        if self.scaled_data is None:
            raise ValueError("Data must be preprocessed first")
        
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            random_state=random_state
        )
        labels = gmm.fit_predict(self.scaled_data)
        
        silhouette = silhouette_score(self.scaled_data, labels)
        davies_bouldin = davies_bouldin_score(self.scaled_data, labels)
        calinski = calinski_harabasz_score(self.scaled_data, labels)
        
        self.cluster_labels = labels
        
        return {
            'labels': labels,
            'n_clusters': n_components,
            'silhouette_score': silhouette,
            'davies_bouldin_score': davies_bouldin,
            'calinski_harabasz_score': calinski,
            'bic': gmm.bic(self.scaled_data),
            'aic': gmm.aic(self.scaled_data),
            'algorithm': 'GMM'
        }
    
    def compare_algorithms(self, algorithm_results):
        comparison_data = []
        
        for algo_name, result in algorithm_results.items():
            row = {
                'Algorithm': algo_name,
                'N_Clusters': result.get('n_clusters', 'N/A'),
                'Silhouette': result.get('silhouette_score', 'N/A'),
                'Davies-Bouldin': result.get('davies_bouldin_score', 'N/A'),
                'Calinski-Harabasz': result.get('calinski_harabasz_score', 'N/A')
            }
            
            if algo_name == 'DBSCAN':
                row['Noise Points'] = result.get('n_noise', 0)
            
            if algo_name == 'GMM':
                row['BIC'] = result.get('bic', 'N/A')
                row['AIC'] = result.get('aic', 'N/A')
            
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
