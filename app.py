import streamlit as st
import pandas as pd
import numpy as np
from clustering import SoilClusteringEngine
from visualization import (
    create_elbow_plot,
    create_3d_cluster_plot,
    create_2d_cluster_plot,
    create_correlation_heatmap,
    create_feature_distribution_plots,
    create_cluster_comparison_plot,
    create_variance_explained_plot
)
import io


st.set_page_config(
    page_title="Precision Farming - Soil Clustering",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.title("🌾 Precision Farming: Soil & Environmental Data Clustering")
st.markdown("""
This application performs multivariate clustering analysis on soil and environmental data 
to support precision farming decisions. Upload your dataset and discover patterns in your agricultural data.
""")


if 'clustering_engine' not in st.session_state:
    st.session_state.clustering_engine = SoilClusteringEngine()
if 'data' not in st.session_state:
    st.session_state.data = None
if 'feature_columns' not in st.session_state:
    st.session_state.feature_columns = None
if 'clustering_done' not in st.session_state:
    st.session_state.clustering_done = False


with st.sidebar:
    st.header("📁 Data Upload")
    uploaded_file = st.file_uploader(
        "Upload your soil/environmental data",
        type=['csv', 'xlsx', 'xls'],
        help="Upload a CSV or Excel file with soil and environmental measurements"
    )
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.session_state.data = df
            st.success(f"✅ Loaded {len(df)} rows and {len(df.columns)} columns")
            
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
    
    st.markdown("---")
    st.markdown("""
    ### 📊 Sample Data Format
    Your data should include columns like:
    - Soil pH
    - Nitrogen (N)
    - Phosphorus (P)
    - Potassium (K)
    - Organic Matter
    - Moisture
    - Temperature
    - etc.
    """)


if st.session_state.data is not None:
    df = st.session_state.data
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Exploration", "🔍 Clustering Analysis", "📈 Results & Insights", "💾 Export"])
    
    with tab1:
        st.header("Data Overview")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Samples", len(df))
        with col2:
            st.metric("Total Features", len(df.columns))
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())
        
        st.subheader("Data Preview")
        st.dataframe(df.head(20), use_container_width=True)
        
        st.subheader("Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)
        
        st.subheader("Select Features for Clustering")
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_columns) > 0:
            selected_features = st.multiselect(
                "Choose numerical features to use for clustering:",
                options=numeric_columns,
                default=numeric_columns[:min(5, len(numeric_columns))],
                help="Select at least 2 features for clustering analysis"
            )
            
            if len(selected_features) >= 2:
                st.session_state.feature_columns = selected_features
                
                st.subheader("Feature Distributions")
                fig_dist = create_feature_distribution_plots(df, selected_features)
                st.plotly_chart(fig_dist, use_container_width=True)
                
                st.subheader("Feature Correlations")
                fig_corr = create_correlation_heatmap(df, selected_features)
                st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.warning("⚠️ Please select at least 2 features for clustering analysis")
        else:
            st.error("❌ No numeric columns found in the dataset")
    
    with tab2:
        st.header("Clustering Analysis")
        
        if st.session_state.feature_columns and len(st.session_state.feature_columns) >= 2:
            
            engine = st.session_state.clustering_engine
            
            st.subheader("Step 1: Data Preprocessing & PCA")
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.info(f"📌 Selected {len(st.session_state.feature_columns)} features for analysis")
                st.write("Features:", ", ".join(st.session_state.feature_columns))
            
            with col2:
                n_pca_components = st.number_input(
                    "PCA Components",
                    min_value=2,
                    max_value=min(3, len(st.session_state.feature_columns)),
                    value=min(3, len(st.session_state.feature_columns)),
                    help="Number of principal components to retain"
                )
            
            if st.button("🔄 Preprocess Data & Apply PCA", type="primary"):
                with st.spinner("Processing data..."):
                    engine.preprocess_data(df, st.session_state.feature_columns)
                    pca_result = engine.apply_pca(n_components=n_pca_components)
                    st.session_state.pca_result = pca_result
                    st.success("✅ Data preprocessed and PCA applied successfully!")
            
            if hasattr(st.session_state, 'pca_result'):
                pca_result = st.session_state.pca_result
                
                st.subheader("PCA Variance Explained")
                fig_var = create_variance_explained_plot(pca_result)
                st.plotly_chart(fig_var, use_container_width=True)
                
                total_variance = pca_result['cumulative_variance'][-1] * 100
                st.info(f"📊 {n_pca_components} components explain {total_variance:.2f}% of total variance")
                
                st.markdown("---")
                st.subheader("Step 2: Determine Optimal Number of Clusters")
                
                max_clusters = st.slider(
                    "Maximum clusters to evaluate",
                    min_value=5,
                    max_value=min(15, len(df) - 1),
                    value=10,
                    help="Calculate metrics for cluster numbers from 2 to this value"
                )
                
                if st.button("📊 Calculate Elbow Metrics", type="primary"):
                    with st.spinner("Calculating optimal cluster metrics..."):
                        elbow_data = engine.calculate_elbow_scores(max_clusters=max_clusters)
                        st.session_state.elbow_data = elbow_data
                        st.success("✅ Elbow analysis completed!")
                
                if hasattr(st.session_state, 'elbow_data'):
                    elbow_data = st.session_state.elbow_data
                    fig_elbow = create_elbow_plot(elbow_data)
                    st.plotly_chart(fig_elbow, use_container_width=True)
                    
                    st.info("""
                    **How to choose optimal k:**
                    - **Elbow Method**: Look for the "elbow" where inertia starts decreasing slowly
                    - **Silhouette Score**: Higher is better (closer to 1)
                    - **Davies-Bouldin Index**: Lower is better (closer to 0)
                    """)
                    
                    best_silhouette_idx = np.argmax(elbow_data['silhouette_scores'])
                    suggested_k = elbow_data['k_values'][best_silhouette_idx]
                    st.success(f"💡 Suggested optimal clusters based on Silhouette Score: **{suggested_k}**")
                    
                    st.markdown("---")
                    st.subheader("Step 3: Perform Clustering")
                    
                    n_clusters = st.number_input(
                        "Number of clusters (k)",
                        min_value=2,
                        max_value=max_clusters,
                        value=suggested_k,
                        help="Select the number of clusters for K-Means algorithm"
                    )
                    
                    if st.button("🎯 Run K-Means Clustering", type="primary"):
                        with st.spinner(f"Performing clustering with {n_clusters} clusters..."):
                            cluster_result = engine.perform_clustering(n_clusters)
                            st.session_state.cluster_result = cluster_result
                            st.session_state.n_clusters = n_clusters
                            st.session_state.clustering_done = True
                            st.success(f"✅ Successfully created {n_clusters} clusters!")
                            st.balloons()
        else:
            st.warning("⚠️ Please select features in the 'Data Exploration' tab first")
    
    with tab3:
        st.header("Clustering Results & Insights")
        
        if st.session_state.clustering_done:
            engine = st.session_state.clustering_engine
            cluster_result = st.session_state.cluster_result
            pca_result = st.session_state.pca_result
            
            st.subheader("Clustering Performance Metrics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Number of Clusters", st.session_state.n_clusters)
            with col2:
                st.metric("Silhouette Score", f"{cluster_result['silhouette_score']:.3f}")
            with col3:
                st.metric("Davies-Bouldin Index", f"{cluster_result['davies_bouldin_score']:.3f}")
            
            st.markdown("---")
            st.subheader("Cluster Visualizations")
            
            if pca_result['pca_data'].shape[1] >= 3:
                st.write("**3D Cluster Plot (PCA Space)**")
                fig_3d = create_3d_cluster_plot(
                    pca_result['pca_data'],
                    cluster_result['labels'],
                    "3D Cluster Visualization"
                )
                st.plotly_chart(fig_3d, use_container_width=True)
            
            st.write("**2D Cluster Plot (First 2 Principal Components)**")
            fig_2d = create_2d_cluster_plot(
                pca_result['pca_data'],
                cluster_result['labels'],
                "2D Cluster Visualization"
            )
            st.plotly_chart(fig_2d, use_container_width=True)
            
            st.markdown("---")
            st.subheader("Cluster Statistics & Profiling")
            
            cluster_stats_data = engine.get_cluster_statistics(df, st.session_state.feature_columns)
            st.session_state.cluster_stats = cluster_stats_data
            
            st.write("**Cluster Sizes**")
            cluster_sizes_df = pd.DataFrame({
                'Cluster': cluster_stats_data['sizes'].index,
                'Number of Samples': cluster_stats_data['sizes'].values,
                'Percentage': (cluster_stats_data['sizes'].values / len(df) * 100).round(2)
            })
            st.dataframe(cluster_sizes_df, use_container_width=True)
            
            st.write("**Cluster Profile Comparison**")
            fig_radar = create_cluster_comparison_plot(
                cluster_stats_data['statistics'],
                st.session_state.feature_columns
            )
            st.plotly_chart(fig_radar, use_container_width=True)
            
            st.write("**Detailed Cluster Statistics**")
            for cluster_id in sorted(cluster_stats_data['df_with_clusters']['Cluster'].unique()):
                with st.expander(f"📊 Cluster {cluster_id} Details"):
                    cluster_data = cluster_stats_data['statistics'].loc[cluster_id]
                    st.dataframe(cluster_data, use_container_width=True)
            
            st.markdown("---")
            st.subheader("Feature Importance (PCA)")
            
            importance_data = engine.get_feature_importance()
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**PCA Components**")
                st.dataframe(importance_data['components'].round(3), use_container_width=True)
            
            with col2:
                st.write("**Feature Contribution**")
                st.dataframe(importance_data['importance'].round(3), use_container_width=True)
        else:
            st.info("👈 Please complete the clustering analysis in the 'Clustering Analysis' tab first")
    
    with tab4:
        st.header("Export Results")
        
        if st.session_state.clustering_done:
            st.subheader("Download Clustered Data")
            
            cluster_stats_data = st.session_state.cluster_stats
            df_with_clusters = cluster_stats_data['df_with_clusters']
            
            csv_buffer = io.StringIO()
            df_with_clusters.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()
            
            st.download_button(
                label="📥 Download Data with Cluster Assignments (CSV)",
                data=csv_data,
                file_name="clustered_soil_data.csv",
                mime="text/csv"
            )
            
            st.subheader("Download Cluster Statistics")
            
            cluster_stats_df = st.session_state.cluster_stats['statistics']
            stats_buffer = io.StringIO()
            cluster_stats_df.to_csv(stats_buffer)
            stats_data = stats_buffer.getvalue()
            
            st.download_button(
                label="📥 Download Cluster Statistics (CSV)",
                data=stats_data,
                file_name="cluster_statistics.csv",
                mime="text/csv"
            )
            
            st.subheader("Summary Report")
            
            summary_text = f"""
# Precision Farming Clustering Analysis Report

## Dataset Information
- Total Samples: {len(df)}
- Features Analyzed: {', '.join(st.session_state.feature_columns)}
- Number of Clusters: {st.session_state.n_clusters}

## Performance Metrics
- Silhouette Score: {st.session_state.cluster_result['silhouette_score']:.4f}
- Davies-Bouldin Index: {st.session_state.cluster_result['davies_bouldin_score']:.4f}
- Inertia: {st.session_state.cluster_result['inertia']:.4f}

## PCA Information
- Components Used: {st.session_state.pca_result['pca_data'].shape[1]}
- Total Variance Explained: {st.session_state.pca_result['cumulative_variance'][-1] * 100:.2f}%

## Cluster Distribution
"""
            
            for cluster_id, size in st.session_state.cluster_stats['sizes'].items():
                percentage = (size / len(df) * 100)
                summary_text += f"- Cluster {cluster_id}: {size} samples ({percentage:.2f}%)\n"
            
            st.download_button(
                label="📥 Download Summary Report (TXT)",
                data=summary_text,
                file_name="clustering_report.txt",
                mime="text/plain"
            )
            
        else:
            st.info("👈 Please complete the clustering analysis first to enable exports")

else:
    st.info("👈 Please upload your soil and environmental data using the sidebar to begin analysis")
    
    st.markdown("---")
    st.subheader("📖 How to Use This Application")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Step 1: Upload Data
        - Upload CSV or Excel file with soil/environmental measurements
        - Ensure your data has numerical columns (pH, nutrients, moisture, etc.)
        
        ### Step 2: Explore Data
        - Review statistical summaries
        - Select features for clustering
        - Examine feature distributions and correlations
        """)
    
    with col2:
        st.markdown("""
        ### Step 3: Cluster Analysis
        - Apply PCA for dimensionality reduction
        - Use elbow method to find optimal cluster count
        - Run K-Means clustering algorithm
        
        ### Step 4: Results & Export
        - Visualize clusters in 2D/3D space
        - Analyze cluster statistics and profiles
        - Download results for further use
        """)
