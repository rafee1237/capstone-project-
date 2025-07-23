import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# App title
st.title("Credit Card Usage Segmentation")

# Upload CSV file
uploaded_file = st.file_uploader("Upload Customer Data CSV", type="csv")
if uploaded_file:
    df = pd.read_csv("C:\\Users\\rafee\\Downloads\\Capstone Project Credit_Card Usage Segmentation\\Customer Data.csv")
    st.write("### Raw Data", df.head())

    # Preprocessing
    features = df.select_dtypes(include=['float64', 'int64']).dropna(axis=1)
    st.write("### Features used for clustering", features.columns.tolist())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    # Elbow method to determine k
    if st.checkbox("Show Elbow Plot"):
        sse = []
        K = range(1, 11)
        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=0)
            kmeans.fit(X_scaled)
            sse.append(kmeans.inertia_)

        plt.figure()
        plt.plot(K, sse, 'bo-')
        plt.xlabel('Number of clusters')
        plt.ylabel('SSE')
        plt.title('Elbow Method For Optimal k')
        st.pyplot(plt)

    # Choose number of clusters
    k = st.slider("Select number of clusters (k)", 2, 10, 3)
    kmeans = KMeans(n_clusters=k, random_state=0)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    st.write("### Clustered Data", df.head())

    # Visualization
    if st.checkbox("Show Cluster Plot"):
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        components = pca.fit_transform(X_scaled)
        df['PCA1'] = components[:, 0]
        df['PCA2'] = components[:, 1]

        plt.figure()
        sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=df, palette='Set2')
        plt.title('Customer Segmentation')
        st.pyplot(plt)
