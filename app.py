import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.decomposition import PCA
import plotly.graph_objs as go
from scipy.spatial.distance import pdist, squareform

st.set_page_config(layout='wide')

@st.cache_resource
def load_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def calculate_angle(p1, p2, p3):
    vec1 = np.array(p2) - np.array(p1)
    vec2 = np.array(p3) - np.array(p2)
    
    cosine_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def analyze_shape(embeddings_2d):
    num_points = len(embeddings_2d)
    
    if num_points < 3:
        return None
    
    angles = [
        calculate_angle(
            embeddings_2d[i - 1], 
            embeddings_2d[i], 
            embeddings_2d[(i + 1) % num_points]
        ) for i in range(num_points)
    ]
    
    rounded_angles = np.round(angles, decimals=2)
    unique_angles = np.unique(rounded_angles)
    
    if num_points == 3:
        if len(unique_angles) == 1:
            return "You drew an equilateral triangle!"
        elif len(unique_angles) == 2:
            return "You drew an isosceles triangle!"
        else:
            return "You drew a scalene triangle!"
    
    elif num_points == 4:
        if len(unique_angles) == 1 and np.isclose(unique_angles[0], 90):
            return "Wow, you drew a square!"
        elif len(unique_angles) == 2 and np.isclose(unique_angles, [90, 90]).all():
            return "You drew a rectangle!"
        elif len(unique_angles) == 1:
            return "You drew a rhombus!"
        else:
            return "You drew a quadrilateral!"
    
    elif len(unique_angles) == 1:
        return f"You drew a regular {num_points}-sided polygon!"
    else:
        return f"You drew an irregular {num_points}-sided shape!"

model = load_model()

if 'texts' not in st.session_state:
    st.session_state['texts'] = []
if 'embeddings' not in st.session_state:
    st.session_state['embeddings'] = []
if 'rounding_precision' not in st.session_state:
    st.session_state['rounding_precision'] = 2
if 'messages' not in st.session_state:
    st.session_state['messages'] = []  
if 'shape' not in st.session_state:
    st.session_state['shape'] = None
if 'show_ai_message' not in st.session_state:
    st.session_state['show_ai_message'] = False

st.title("Words to Shapes!")
st.write("Type a short text or phrase into the chat input to add a point representing its semantic embedding on a 2D plot. Add multiple texts to see how the shape evolves and what the app determines about its geometry!")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Chat Input")
    
    messages = st.container()
    with messages:
        st.chat_message("ai").write("Hi! Will you please help me learn your language? To get started, please write three different words or phrases to draw a triangle.")
        for message in st.session_state['messages']:
            st.chat_message("user").write(message)
        if st.session_state['shape']:
            st.chat_message("ai").write(st.session_state['shape'])

    if prompt := st.chat_input("Enter text to analyze as a shape point"):
        if prompt not in st.session_state['texts']:
            st.session_state['texts'].append(prompt)
            st.session_state['messages'].append(prompt)
            embedding = model.encode(prompt, convert_to_tensor=False)
            st.session_state['embeddings'].append(embedding)
            st.rerun()

with col2:
    if len(st.session_state['embeddings']) > 0:
        embeddings = np.array(st.session_state['embeddings'])

        if len(embeddings) > 1:
            pca = PCA(n_components=2)
            embeddings_2d = pca.fit_transform(embeddings)
        else:
            embeddings_2d = np.hstack([embeddings, np.zeros((len(embeddings), 1))])

        embeddings_2d = np.round(embeddings_2d, decimals=st.session_state['rounding_precision'])

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
        if shape_description and shape_description != st.session_state.get('shape', ''):
            st.session_state['shape'] = shape_description
            st.session_state['show_ai_message'] = True
            st.rerun()

if st.session_state['show_ai_message']:
    st.session_state['show_ai_message'] = False
