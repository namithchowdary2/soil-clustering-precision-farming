import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def create_elbow_plot(elbow_data):
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Elbow Method (Inertia)', 'Silhouette Score', 'Davies-Bouldin Index'),
        horizontal_spacing=0.1
    )
    
    fig.add_trace(
        go.Scatter(
            x=elbow_data['k_values'],
            y=elbow_data['inertias'],
            mode='lines+markers',
            name='Inertia',
            line=dict(color='blue', width=2),
            marker=dict(size=8)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=elbow_data['k_values'],
            y=elbow_data['silhouette_scores'],
            mode='lines+markers',
            name='Silhouette',
            line=dict(color='green', width=2),
            marker=dict(size=8)
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=elbow_data['k_values'],
            y=elbow_data['davies_bouldin_scores'],
            mode='lines+markers',
            name='Davies-Bouldin',
            line=dict(color='red', width=2),
            marker=dict(size=8)
        ),
        row=1, col=3
    )
    
    fig.update_xaxes(title_text="Number of Clusters (k)", row=1, col=1)
    fig.update_xaxes(title_text="Number of Clusters (k)", row=1, col=2)
    fig.update_xaxes(title_text="Number of Clusters (k)", row=1, col=3)
    
    fig.update_yaxes(title_text="Inertia", row=1, col=1)
    fig.update_yaxes(title_text="Score (higher is better)", row=1, col=2)
    fig.update_yaxes(title_text="Score (lower is better)", row=1, col=3)
    
    fig.update_layout(
        height=400,
        showlegend=False,
        title_text="Optimal Cluster Selection Metrics"
    )
    
    return fig


def create_3d_cluster_plot(pca_data, labels, title="3D Cluster Visualization (PCA)"):
    df_plot = pd.DataFrame({
        'PC1': pca_data[:, 0],
        'PC2': pca_data[:, 1],
        'PC3': pca_data[:, 2],
        'Cluster': labels.astype(str)
    })
    
    fig = px.scatter_3d(
        df_plot,
        x='PC1',
        y='PC2',
        z='PC3',
        color='Cluster',
        title=title,
        labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2', 'PC3': 'Principal Component 3'},
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    
    fig.update_traces(marker=dict(size=5, line=dict(width=0.5, color='white')))
    fig.update_layout(height=600)
    
    return fig


def create_2d_cluster_plot(pca_data, labels, title="2D Cluster Visualization (PCA)"):
    df_plot = pd.DataFrame({
        'PC1': pca_data[:, 0],
        'PC2': pca_data[:, 1],
        'Cluster': labels.astype(str)
    })
    
    fig = px.scatter(
        df_plot,
        x='PC1',
        y='PC2',
        color='Cluster',
        title=title,
        labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2'},
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    
    fig.update_traces(marker=dict(size=8, line=dict(width=1, color='white')))
    fig.update_layout(height=500)
    
    return fig


def create_correlation_heatmap(df, feature_columns):
    correlation_matrix = df[feature_columns].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=correlation_matrix.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title="Feature Correlation Heatmap",
        height=600,
        xaxis_title="Features",
        yaxis_title="Features"
    )
    
    return fig


def create_feature_distribution_plots(df, feature_columns):
    n_features = len(feature_columns)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=feature_columns,
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    for idx, feature in enumerate(feature_columns):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        
        fig.add_trace(
            go.Histogram(
                x=df[feature],
                name=feature,
                showlegend=False,
                marker_color='steelblue'
            ),
            row=row,
            col=col
        )
    
    fig.update_layout(
        height=300 * n_rows,
        title_text="Feature Distributions",
        showlegend=False
    )
    
    return fig


def create_cluster_comparison_plot(cluster_stats_df, feature_columns):
    cluster_means = cluster_stats_df.xs('mean', axis=1, level=1)[feature_columns]
    
    fig = go.Figure()
    
    for cluster in cluster_means.index:
        fig.add_trace(go.Scatterpolar(
            r=cluster_means.loc[cluster].values,
            theta=feature_columns,
            fill='toself',
            name=f'Cluster {cluster}'
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[cluster_means.min().min() * 0.9, cluster_means.max().max() * 1.1]
            )
        ),
        showlegend=True,
        title="Cluster Profile Comparison (Mean Values)",
        height=600
    )
    
    return fig


def create_variance_explained_plot(variance_data):
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Variance Explained by Component', 'Cumulative Variance Explained'),
        horizontal_spacing=0.15
    )
    
    components = [f'PC{i+1}' for i in range(len(variance_data['variance_explained']))]
    
    fig.add_trace(
        go.Bar(
            x=components,
            y=variance_data['variance_explained'] * 100,
            name='Variance Explained',
            marker_color='steelblue'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=components,
            y=variance_data['cumulative_variance'] * 100,
            mode='lines+markers',
            name='Cumulative',
            line=dict(color='red', width=2),
            marker=dict(size=8)
        ),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text="Principal Component", row=1, col=1)
    fig.update_xaxes(title_text="Principal Component", row=1, col=2)
    fig.update_yaxes(title_text="Variance Explained (%)", row=1, col=1)
    fig.update_yaxes(title_text="Cumulative Variance (%)", row=1, col=2)
    
    fig.update_layout(height=400, showlegend=False)
    
    return fig
