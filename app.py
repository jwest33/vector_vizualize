import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.decomposition import PCA
import plotly.graph_objs as go

@st.cache_resource
def load_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

model = load_model()

if 'texts' not in st.session_state:
    st.session_state['texts'] = []
if 'embeddings' not in st.session_state:
    st.session_state['embeddings'] = []

st.title("3D Text Embedding Visualization")

text_input = st.text_input("Enter text:")

if text_input:
    st.session_state['texts'].append(text_input)
    
    embedding = model.encode(text_input, convert_to_tensor=False)
    st.session_state['embeddings'].append(embedding)

if len(st.session_state['embeddings']) > 0:
    # Reduce to 3 dimensions using PCA
    embeddings = np.array(st.session_state['embeddings'])
    pca = PCA(n_components=3)
    embeddings_3d = pca.fit_transform(embeddings)

    # Create a 3D plot using Plotly
    trace = go.Scatter3d(
        x=embeddings_3d[:, 0], 
        y=embeddings_3d[:, 1], 
        z=embeddings_3d[:, 2],
        mode='markers+text',
        text=st.session_state['texts'],
        marker=dict(
            size=10,
            color=np.arange(len(st.session_state['texts'])),
            colorscale='Viridis',
            opacity=0.8
        )
    )

    layout = go.Layout(
        title="3D Plot of Text Embeddings",
        scene=dict(
            xaxis_title="PCA Component 1",
            yaxis_title="PCA Component 2",
            zaxis_title="PCA Component 3"
        ),
        margin=dict(l=0, r=0, b=0, t=0)
    )

    fig = go.Figure(data=[trace], layout=layout)

    st.plotly_chart(fig)

    st.subheader("3D Coordinates of Texts:")
    for i, text in enumerate(st.session_state['texts']):
        st.write(f"Text: {text}")
        st.write(f"3D Coordinates: {embeddings_3d[i]}")
