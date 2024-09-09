import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.decomposition import PCA
import plotly.graph_objs as go

# Load the pre-trained model (you can use 'e5-base-v2' or any similar model)
@st.cache_resource
def load_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

model = load_model()

# Initialize session state for storing embeddings and texts
if 'texts' not in st.session_state:
    st.session_state['texts'] = []
if 'embeddings' not in st.session_state:
    st.session_state['embeddings'] = []

# Title
st.title("3D Text Embedding Visualization")

# Text input
text_input = st.text_input("Enter text:")

# Process the text
if text_input:
    st.session_state['texts'].append(text_input)
    
    # Generate embeddings
    embedding = model.encode(text_input, convert_to_tensor=False)
    st.session_state['embeddings'].append(embedding)
    
    # Notify the user
    st.success(f'Text "{text_input}" has been processed and embedded!')

# Only proceed if we have at least one embedding
if len(st.session_state['embeddings']) > 0:
    embeddings_2d = np.array(st.session_state['embeddings'])

    if len(embeddings_2d) == 1:
        # Case 1: Only one text - manually reduce to 3 dimensions by truncating
        embedding_3d = embeddings_2d[0][:3]  # Take the first 3 dimensions directly
        embeddings_3d = np.array([embedding_3d])  # Convert to a 2D array with one row
    elif len(embeddings_2d) >= 3:
        # Case 2: Apply PCA if there are 3 or more texts
        pca = PCA(n_components=3)
        embeddings_3d = pca.fit_transform(embeddings_2d)
    else:
        # Case 3: Two texts - reduce to 2 dimensions with PCA
        pca = PCA(n_components=2)
        embeddings_2d_pca = pca.fit_transform(embeddings_2d)
        embeddings_3d = np.hstack([embeddings_2d_pca, np.zeros((embeddings_2d_pca.shape[0], 1))])  # Add a zero z-axis

    # Create a 3D plot using Plotly
    trace = go.Scatter3d(
        x=embeddings_3d[:, 0], 
        y=embeddings_3d[:, 1], 
        z=embeddings_3d[:, 2],
        mode='markers+text',
        text=st.session_state['texts'],
        marker=dict(
            size=10,
            color=np.arange(len(st.session_state['texts'])),  # Assign a unique color per point
            colorscale='Viridis',
            opacity=0.8
        )
    )

    layout = go.Layout(
        title="3D Plot of Text Embeddings",
        scene=dict(
            xaxis_title="Dimension 1",
            yaxis_title="Dimension 2",
            zaxis_title="Dimension 3"
        ),
        margin=dict(l=0, r=0, b=0, t=0)
    )

    fig = go.Figure(data=[trace], layout=layout)

    # Display the plot
    st.plotly_chart(fig)

    # Display the raw 3D coordinates
    st.subheader("3D Coordinates of Texts:")
    for i, text in enumerate(st.session_state['texts']):
        st.write(f"Text: {text}")
        st.write(f"3D Coordinates: {embeddings_3d[i]}")
