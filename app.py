import streamlit as st
import numpy as np
import cv2
from PIL import Image
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained model
with open('brain_tumor_model.pkl', 'rb') as file:
    model = pickle.load(file)

st.markdown("""
<style>
.big-font {
    font-size:30px !important;
    font-weight: bold;
}
.medium-font {
    font-size:20px !important;
}
.stButton>button {
    color: #4F8BF9;
    border-radius: 50px;
    height: 3em;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

def preprocess_image(image):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (200, 200))
    img = img.reshape(1, -1) / 255.0
    return img

def predict(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    decision_function = model.decision_function(processed_image)
    confidence = 1 / (1 + np.exp(-decision_function[0]))  # Convert to probability-like score
    return "Tumor Detected" if prediction[0] == 1 else "No Tumor Detected", confidence

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Predict", "About", "Statistics"])

    if page == "Home":
        display_home()
    elif page == "Predict":
        display_predict()
    elif page == "About":
        display_about()
    elif page == "Statistics":
        display_statistics()

def display_home():
    st.markdown('<p class="big-font">Welcome to Brain Tumor Detection</p>', unsafe_allow_html=True)
    st.markdown('<p class="medium-font">This application uses machine learning to detect brain tumors from MRI scans.</p>', unsafe_allow_html=True)
    st.image("download.jpg", use_column_width=True)
    st.markdown("### How to use this app:")
    st.write("1. Navigate to the 'Predict' page using the sidebar")
    st.write("2. Upload an MRI scan image")
    st.write("3. Click 'Predict' to get the results")
    st.write("4. View detailed statistics on the 'Statistics' page")

def display_predict():
    st.markdown('<p class="big-font">Brain Tumor Detection</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose an MRI scan...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded MRI scan.', use_column_width=True)
        
        if st.button('Predict'):
            with st.spinner('Analyzing the image...'):
                label, confidence = predict(image)
            
            st.success(f"Prediction: {label}")
            st.write(f"Confidence: {confidence:.2%}")
            
            # Display a gauge chart for the confidence
            fig, ax = plt.subplots()
            sns.barplot(x=['No Tumor', 'Tumor'], y=[1-confidence, confidence], ax=ax)
            ax.set_ylim(0, 1)
            ax.set_title('Prediction Confidence')
            st.pyplot(fig)

def display_about():
    st.markdown('<p class="big-font">About This Project</p>', unsafe_allow_html=True)
    st.write("""
    This brain tumor detection project uses machine learning to analyze MRI scans and predict the presence of tumors. 
    The model was trained on a dataset of brain MRI scans, including both tumor and non-tumor images.
    
    Key features of this project:
    - Utilizes a Support Vector Machine (SVM) classifier
    - Preprocesses images using OpenCV
    - Provides confidence scores for predictions
    - Built with Streamlit for an interactive web interface
    
    """)

def display_statistics():
    st.markdown('<p class="big-font">Model Statistics</p>', unsafe_allow_html=True)
    
    # You should replace these with actual statistics from your model
    accuracy = 0.9633
    precision = 0.9632
    recall = 0.9633
    f1_score = 0.9631
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(label="Accuracy", value=f"{accuracy:.2%}")
        st.metric(label="Precision", value=f"{precision:.2%}")
    
    with col2:
        st.metric(label="Recall", value=f"{recall:.2%}")
        st.metric(label="F1 Score", value=f"{f1_score:.2%}")
    
    # Display a dummy confusion matrix
    confusion_matrix = np.array([[190, 10], [5, 200]])
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    st.pyplot(fig)

if __name__ == "__main__":
    main()