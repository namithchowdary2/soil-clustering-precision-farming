# 🌾 Precision Farming: Soil & Environmental Data Clustering

A comprehensive machine learning application for analyzing soil and environmental data to support precision farming decisions through multivariate clustering analysis.

## 🚀 Quick Start

### Option 1: Web Application (Already Running!)
The Streamlit web app is **live and running** on this Repl. Simply:
1. Click the webview to open the application
2. Upload your CSV file using the sidebar
3. Start analyzing your data immediately!

### Option 2: Google Colab Notebook
1. Download the `Soil_Clustering_Analysis_Colab.ipynb` file
2. Upload it to Google Colab: https://colab.research.google.com/
3. Run all cells and upload your CSV when prompted
4. Follow the step-by-step analysis

## 📊 Features

### Core Functionality
- **Data Upload**: Support for CSV and Excel files
- **Data Exploration**: Interactive visualizations, correlations, and statistical summaries
- **Multiple Clustering Algorithms**:
  - K-Means with elbow method optimization
  - DBSCAN (Density-based clustering)
  - Hierarchical Clustering (Agglomerative)
  - Gaussian Mixture Models (GMM)
- **PCA Dimensionality Reduction**: Visualize high-dimensional data in 2D/3D
- **Algorithm Comparison**: Side-by-side metrics to choose the best approach
- **Cluster Profiling**: Detailed statistics and insights per cluster
- **Export Results**: Download clustered data and analysis reports

### Visualizations
- 3D and 2D scatter plots of clusters
- Correlation heatmaps
- Feature distribution plots
- Elbow method charts
- PCA variance explained plots
- Radar charts for cluster comparison

## 📁 Project Structure

```
├── app.py                          # Main Streamlit web application
├── clustering.py                   # Backend clustering engine with all ML algorithms
├── visualization.py                # Plotly/Matplotlib visualization functions
├── sample_soil_data.csv           # Sample dataset for testing
├── Soil_Clustering_Analysis_Colab.ipynb  # Google Colab notebook version
└── README.md                      # This file
```

## 🔧 Backend Code (clustering.py)

The `clustering.py` file contains the `SoilClusteringEngine` class with all the machine learning functionality:

```python
from clustering import SoilClusteringEngine

# Initialize
engine = SoilClusteringEngine()

# Preprocess data
engine.preprocess_data(df, feature_columns)

# Apply PCA
pca_result = engine.apply_pca(n_components=3)

# Run clustering
kmeans_result = engine.perform_kmeans(n_clusters=3)
dbscan_result = engine.perform_dbscan(eps=0.5, min_samples=5)
hierarchical_result = engine.perform_hierarchical(n_clusters=3)
gmm_result = engine.perform_gmm(n_components=3)

# Get statistics
stats = engine.get_cluster_statistics(df, feature_columns, kmeans_result['labels'])
```

## 📊 Using Your Own Data

Your CSV file should contain numerical columns representing soil/environmental measurements. Example columns:
- `pH` - Soil acidity/alkalinity
- `Nitrogen_ppm` - Nitrogen content (parts per million)
- `Phosphorus_ppm` - Phosphorus content
- `Potassium_ppm` - Potassium content
- `Organic_Matter_percent` - Organic matter percentage
- `Moisture_percent` - Soil moisture
- `Temperature_C` - Soil temperature
- `EC_dS_m` - Electrical conductivity

A sample dataset (`sample_soil_data.csv`) is included for testing.

## 🧪 Running on Google Colab

The notebook includes:
1. **Automatic package installation** - All dependencies installed automatically
2. **File upload widget** - Easy CSV upload interface
3. **Step-by-step workflow** - Clear sections with explanations
4. **All 4 algorithms** - K-Means, DBSCAN, Hierarchical, GMM
5. **Interactive visualizations** - 3D plots, comparisons, heatmaps
6. **Download results** - Export clustered data and statistics

Just upload the `.ipynb` file to Colab and run all cells!

## 📦 Dependencies

```python
pandas
numpy
scikit-learn
plotly
matplotlib
seaborn
streamlit
openpyxl
```

All dependencies are already installed in this Repl and auto-installed in the Colab notebook.

## 🎯 Use Cases

- **Zone Management**: Identify management zones in agricultural fields
- **Variable Rate Application**: Optimize fertilizer/irrigation based on soil clusters
- **Crop Selection**: Match crops to soil characteristics by cluster
- **Soil Health Monitoring**: Track changes in soil properties over time
- **Precision Agriculture**: Data-driven farming decisions

## 📈 Clustering Metrics Explained

- **Silhouette Score** (0 to 1): Measures cluster separation. Higher is better.
- **Davies-Bouldin Index**: Measures cluster compactness. Lower is better.
- **Calinski-Harabasz Score**: Ratio of between/within cluster dispersion. Higher is better.
- **BIC/AIC** (GMM only): Model selection criteria. Lower is better.

## 🤝 Support

For questions or issues:
1. Check the sample dataset to understand the expected format
2. Review the step-by-step guide in the Colab notebook
3. Use the web app's interactive interface for easier analysis

## 📝 License

This project is open source and available for agricultural research and precision farming applications.

---

**Happy Farming! 🌾**
