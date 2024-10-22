import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.decomposition import PCA
import plotly.graph_objs as go
from scipy.spatial.distance import pdist, squareform

@st.cache_resource
def load_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def analyze_shape(embeddings_2d): # Temporary placeholder for proper geometry analysis and scoring
    distances = pdist(embeddings_2d, 'euclidean')
    dist_matrix = squareform(distances)
    edge_distances = [dist_matrix[i, (i + 1) % len(embeddings_2d)] for i in range(len(embeddings_2d))]
    rounded_distances = np.round(edge_distances, decimals=2)
    unique_distances = np.unique(rounded_distances)
    num_points = len(embeddings_2d)
    
    if num_points < 3:
        return "Not enough points to form a shape."
    if num_points == 3:
        if len(unique_distances) == 1:
            return "You drew an equilateral triangle!"
        elif len(unique_distances) == 2:
            return "You drew an isosceles triangle!"
        else:
            return "You drew a scalene triangle!"
    if num_points == 4:
        if len(unique_distances) == 1:
            return "Wow, you drew a square!"
        elif len(unique_distances) == 2:
            return "You drew a rectangle or a rhombus!"
        else:
            return "You drew a quadrilateral!"
    if len(unique_distances) == 1:
        return f"You drew a regular {num_points}-sided polygon!"
    else:
        return f"You drew an irregular {num_points}-sided shape!"

model = load_model()

if 'texts' not in st.session_state:
    st.session_state['texts'] = []
if 'embeddings' not in st.session_state:
    st.session_state['embeddings'] = []

st.title("Words to Shapes!")
st.write("Type a short text or phrase into the input box to add a point representing its semantic embedding on a 2D plot. Add multiple texts to see how the shape evolves and what the app determines about its geometry!")

st.divider()

text_input = st.text_input("Enter text:")

if text_input:
    st.session_state['texts'].append(text_input)
    
    embedding = model.encode(text_input, convert_to_tensor=False)
    st.session_state['embeddings'].append(embedding)

if len(st.session_state['embeddings']) > 0:
    embeddings = np.array(st.session_state['embeddings'])

    if len(embeddings) > 1:
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)
    else:
        embeddings_2d = np.hstack([embeddings, np.zeros((len(embeddings), 1))])

    trace_nodes = go.Scatter(
        x=embeddings_2d[:, 0], 
        y=embeddings_2d[:, 1], 
        mode='markers+text',
        text=st.session_state['texts'],
        textposition="top center",
        marker=dict(
            size=10,
            color=np.arange(len(st.session_state['texts'])), 
            colorscale='Viridis',
            opacity=0.8
        )
    )
    
    if len(embeddings_2d) > 1:
        x_edges = []
        y_edges = []
        for i in range(1, len(embeddings_2d)):
            x_edges.extend([embeddings_2d[i - 1, 0], embeddings_2d[i, 0], None])
            y_edges.extend([embeddings_2d[i - 1, 1], embeddings_2d[i, 1], None])
        x_edges.extend([embeddings_2d[-1, 0], embeddings_2d[0, 0], None])
        y_edges.extend([embeddings_2d[-1, 1], embeddings_2d[0, 1], None])

        trace_edges = go.Scatter(
            x=x_edges,
            y=y_edges,
            mode='lines',
            line=dict(color='gray', width=1),
            hoverinfo='none'
        )
    else:
        trace_edges = None

    padding = 0.1

    x_min, x_max = embeddings_2d[:, 0].min(), embeddings_2d[:, 0].max()
    y_min, y_max = embeddings_2d[:, 1].min(), embeddings_2d[:, 1].max()

    x_range = [x_min - padding * abs(x_max - x_min), x_max + padding * abs(x_max - x_min)]
    y_range = [y_min - padding * abs(y_max - y_min), y_max + padding * abs(y_max - y_min)]

    layout = go.Layout(
        title=None,
        xaxis=dict(title=None, showgrid=False, zeroline=False, showticklabels=False, range=x_range),
        yaxis=dict(title=None, showgrid=False, zeroline=False, showticklabels=False, range=y_range),
        margin=dict(l=0, r=0, b=0, t=0),
        showlegend=False
    )

    fig = go.Figure(data=[trace_nodes] + ([trace_edges] if trace_edges else []), layout=layout)

    st.plotly_chart(fig)

    shape_description = analyze_shape(embeddings_2d)
    st.subheader("Shape Analysis:")
    st.write(shape_description)
    
    st.divider()

    st.subheader("2D Coordinates of Texts:")
    for i, text in enumerate(st.session_state['texts']):
        st.write(f"Text: {text}")
        st.write(f"2D Coordinates: {embeddings_2d[i]}")
