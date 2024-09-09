# 3D Text Embedding Visualization with Streamlit
This application visualizes text embeddings in 3D space, reducing high-dimensional vector representations (generated using a pre-trained model) down to three dimensions using PCA. Users can input multiple text strings, which are embedded, reduced, and plotted interactively on a 3D graph.

![3D Plot Example](example.png)

## Features
- **Text Embedding**: Converts input text into high-dimensional embeddings using a Sentence Transformer model.
- **Dimensionality Reduction**: Embeddings are reduced from 768 dimensions to 3 using Principal Component Analysis (PCA).
- **Dynamic Plotting**: Multiple text inputs are visualized as 3D points on an interactive Plotly graph. Supports varying numbers of inputs with appropriate handling for 1D, 2D, and 3D scenarios.

## Setup Instructions
### Prerequisites
Ensure the following dependencies are installed:

 - streamlit
 - sentence-transformers
 - scikit-learn
 - plotly
Install the dependencies using pip:

```
bash
Copy code
pip install streamlit sentence-transformers scikit-learn plotly
Running the Application
Clone the repository:
```
```
bash
Copy code
git clone https://github.com/your-username/repository-name.git
cd repository-name
Run the Streamlit app:
```
```
bash
Copy code
streamlit run app.py
Open the provided URL in your browser to interact with the application.
```
## How It Works
1. Enter any text into the input box.
2. The app will generate vector embeddings (typically 768 dimensions).
3. Embeddings are reduced to 3D using PCA (or truncated if necessary).
4. Each input is visualized as a point on a 3D scatter plot.

## Edge Cases
 - **Single Text Input**: The embedding is truncated to 3 dimensions, allowing it to be plotted directly.
 - **Two Text Inputs**: Reduced to 2 dimensions and plotted on a 2D plane, with a z-axis of zero.
 - Three or More Inputs: Standard PCA reduction to 3 dimensions is applied.
