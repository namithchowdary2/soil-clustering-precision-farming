import streamlit as st
import pandas as pd
import numpy as np
import io

# ===== IMPORT PROJECT MODULES =====
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

# ===== STREAMLIT CONFIG =====
st.set_page_config(
    page_title="Precision Farming - Soil Clustering",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌾 Precision Farming: Soil & Environmental Data Clustering")
st.markdown(
    "Unsupervised machine learning for agropedology segmentation and precision farming."
)

# ===== SESSION STATE (SAFE INIT) =====
if "engine" not in st.session_state:
    st.session_state.engine = SoilClusteringEngine()

if "data" not in st.session_state:
    st.session_state.data = None

if "features" not in st.session_state:
    st.session_state.features = []

if "pca_result" not in st.session_state:
    st.session_state.pca_result = None

if "cluster_result" not in st.session_state:
    st.session_state.cluster_result = None

if "cluster_stats" not in st.session_state:
    st.session_state.cluster_stats = None

# ===== SIDEBAR: FILE UPLOAD =====
with st.sidebar:
    st.header("📁 Upload Soil Dataset")

    uploaded_file = st.file_uploader(
        "Upload CSV or Excel file",
        type=["csv", "xlsx", "xls"]
    )

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.session_state.data = df
            st.success(f"Loaded {df.shape[0]} rows × {df.shape[1]} columns")

        except Exception as e:
            st.error(f"File loading error: {e}")

# ⛔ STOP APP SAFELY IF NO DATA
if st.session_state.data is None:
    st.info("👈 Upload a dataset to start analysis")
    st.stop()

df = st.session_state.data

# ===== TABS =====
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Data Exploration",
    "🔍 Clustering Analysis",
    "📈 Results & Insights",
    "💾 Export",
    "🌱 Crop Recommendation"
])

# =====================================================
# TAB 1: DATA EXPLORATION
# =====================================================
with tab1:
    st.subheader("Dataset Overview")
    st.dataframe(df.head(20), use_container_width=True)

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    st.session_state.features = st.multiselect(
        "Select numerical features for clustering",
        numeric_cols,
        default=numeric_cols[:min(5, len(numeric_cols))]
    )

    if len(st.session_state.features) >= 2:
        st.plotly_chart(
            create_feature_distribution_plots(df, st.session_state.features),
            use_container_width=True
        )
        st.plotly_chart(
            create_correlation_heatmap(df, st.session_state.features),
            use_container_width=True
        )
    else:
        st.warning("Select at least two numerical features.")

# =====================================================
# TAB 2: CLUSTERING
# =====================================================
with tab2:
    if len(st.session_state.features) < 2:
        st.warning("Please select features first.")
        st.stop()

    engine = st.session_state.engine

    n_components = st.slider(
        "PCA Components",
        min_value=2,
        max_value=min(3, len(st.session_state.features)),
        value=2
    )

    if st.button("Run PCA"):
        engine.preprocess_data(df, st.session_state.features)
        st.session_state.pca_result = engine.apply_pca(n_components)
        st.success("PCA completed")

    if st.session_state.pca_result is not None:
        st.plotly_chart(
            create_variance_explained_plot(st.session_state.pca_result),
            use_container_width=True
        )

        k = st.slider("Number of clusters (K-Means)", 2, 10, 3)

        if st.button("Run K-Means"):
            st.session_state.cluster_result = engine.perform_clustering(k)
            st.session_state.cluster_stats = engine.get_cluster_statistics(
                df, st.session_state.features
            )
            st.success("Clustering completed")

# =====================================================
# TAB 3: RESULTS
# =====================================================
with tab3:
    if st.session_state.cluster_result is None:
        st.info("Run clustering to see results")
        st.stop()

    result = st.session_state.cluster_result
    pca = st.session_state.pca_result

    col1, col2 = st.columns(2)
    col1.metric("Silhouette Score", round(result["silhouette_score"], 3))
    col2.metric("Davies–Bouldin Index", round(result["davies_bouldin_score"], 3))

    st.plotly_chart(
        create_2d_cluster_plot(pca["pca_data"], result["labels"]),
        use_container_width=True
    )

    if pca["pca_data"].shape[1] >= 3:
        st.plotly_chart(
            create_3d_cluster_plot(pca["pca_data"], result["labels"]),
            use_container_width=True
        )

# =====================================================
# TAB 4: EXPORT
# =====================================================
with tab4:
    if st.session_state.cluster_stats is not None:
        df_out = st.session_state.cluster_stats["df_with_clusters"]

        csv_buffer = io.StringIO()
        df_out.to_csv(csv_buffer, index=False)

        st.download_button(
            "Download Clustered Dataset",
            csv_buffer.getvalue(),
            "clustered_soil_data.csv",
            "text/csv"
        )

# =====================================================
# TAB 5: CROP RECOMMENDATION
# =====================================================
with tab5:
    st.subheader("Crop Recommendation Engine")

    n = st.number_input("Nitrogen (ppm)", 0, 300, 50)
    p = st.number_input("Phosphorus (ppm)", 0, 200, 20)
    k = st.number_input("Potassium (ppm)", 0, 500, 100)
    rainfall = st.number_input("Rainfall (mm)", 0, 5000, 700)
    ph = st.slider("Soil pH", 4.0, 9.0, 6.5)

    if st.button("Recommend Crops"):
        recs = st.session_state.engine.predict_crop(n, p, k, rainfall, ph)
        st.dataframe(
            pd.DataFrame(recs, columns=["Crop", "Suitability (%)"]),
            use_container_width=True
        )
