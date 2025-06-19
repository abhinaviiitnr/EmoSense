import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import plotly.express as px
import plotly.graph_objects as go
from textblob import TextBlob
import re
from datetime import datetime
import time

# Configure Streamlit page
st.set_page_config(
    page_title="EmoSense - Emotion Detection",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional dark gradient theme
st.markdown("""
<style>
    /* Main dark background with subtle gradient */
    .stApp {
        background: linear-gradient(135deg, #0c0c0c 0%, #1a1a2e 25%, #16213e 50%, #0f3460 75%, #0c0c0c 100%);
        background-attachment: fixed;
        color: #ffffff;
    }
    
    /* Main container styling */
    .main .block-container {
        background: rgba(30, 30, 40, 0.95);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Sidebar dark styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #1e1e2e 0%, #252540 100%);
        border-radius: 15px;
        padding: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Header styling */
    h1 {
        background: linear-gradient(90deg, #64ffda, #1de9b6, #00bcd4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        font-size: 3rem !important;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    
    /* Subheader styling */
    h2 {
        color: #e2e8f0;
        font-weight: 600;
        margin-top: 2rem;
        padding-left: 1rem;
        border-left: 4px solid #64ffda;
    }
    
    h3 {
        color: #cbd5e0;
        font-weight: 500;
        margin-top: 1.5rem;
    }
    
    /* Text color */
    .stMarkdown, .stText, p, span, div {
        color: #e2e8f0 !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(45deg, #1de9b6, #00bcd4, #0097a7);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(29, 233, 182, 0.4);
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(29, 233, 182, 0.6);
        background: linear-gradient(45deg, #00bcd4, #1de9b6, #64ffda);
    }
    
    /* Text area and input styling */
    .stTextArea > div > div > textarea {
        background: rgba(45, 45, 60, 0.9) !important;
        border: 2px solid rgba(100, 255, 218, 0.3) !important;
        border-radius: 15px !important;
        padding: 1rem !important;
        color: #e2e8f0 !important;
        backdrop-filter: blur(5px);
    }
    
    .stTextArea > div > div > textarea:focus {
        border: 2px solid #64ffda !important;
        box-shadow: 0 0 15px rgba(100, 255, 218, 0.4) !important;
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div > select {
        background: rgba(45, 45, 60, 0.9) !important;
        border: 2px solid rgba(100, 255, 218, 0.3) !important;
        border-radius: 10px !important;
        padding: 0.5rem !important;
        color: #e2e8f0 !important;
    }
    
    /* Metric cards */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #2d2d3c 0%, #383850 100%);
        border: 1px solid rgba(100, 255, 218, 0.2);
        padding: 1rem;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
    }
    
    [data-testid="metric-container"] > div {
        color: #e2e8f0 !important;
    }
    
    /* Info boxes */
    .stInfo {
        background: linear-gradient(90deg, #1a2332, #2d3748);
        border-left: 4px solid #64ffda;
        border-radius: 10px;
    }
    
    .stSuccess {
        background: linear-gradient(90deg, #1a2e1a, #2d5016);
        border-left: 4px solid #48bb78;
        border-radius: 10px;
    }
    
    .stWarning {
        background: linear-gradient(90deg, #2d2416, #4a3728);
        border-left: 4px solid #ed8936;
        border-radius: 10px;
    }
    
    .stError {
        background: linear-gradient(90deg, #2d1a1a, #4a2c2a);
        border-left: 4px solid #f56565;
        border-radius: 10px;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #1de9b6, #00bcd4);
        border-radius: 10px;
    }
    
    /* Dataframe styling */
    .stDataFrame {
        background: rgba(45, 45, 60, 0.95);
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(100, 255, 218, 0.1);
    }
    
    /* Chat messages */
    .stChatMessage {
        background: rgba(45, 45, 60, 0.9);
        border-radius: 15px;
        margin: 0.5rem 0;
        padding: 1rem;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(100, 255, 218, 0.1);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(90deg, #2d2d3c, #383850);
        border-radius: 10px;
        padding: 0.5rem;
        border: 1px solid rgba(100, 255, 218, 0.2);
        color: #e2e8f0 !important;
    }
    
    /* File uploader */
    .stFileUploader > div {
        background: rgba(45, 45, 60, 0.9);
        border: 2px dashed rgba(100, 255, 218, 0.4);
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        backdrop-filter: blur(5px);
    }
    
    /* Radio buttons */
    .stRadio > div {
        background: rgba(45, 45, 60, 0.8);
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid rgba(100, 255, 218, 0.1);
    }
    
    /* Sidebar header */
    .sidebar .sidebar-content h2 {
        background: linear-gradient(90deg, #64ffda, #1de9b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 600;
        border: none;
        padding: 0;
    }
    
    /* Custom styling for prediction results */
    .prediction-card {
        background: linear-gradient(135deg, #2d2d3c 0%, #383850 100%);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.4);
        border: 1px solid rgba(100, 255, 218, 0.2);
    }
    
    /* Footer styling */
    .footer {
        background: linear-gradient(90deg, #64ffda, #1de9b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        font-weight: 500;
        padding: 1rem;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(45, 45, 60, 0.3);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #64ffda, #1de9b6);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #1de9b6, #00bcd4);
    }
    
    /* Label styling */
    .stSelectbox label, .stTextArea label, .stTextInput label, .stRadio label {
        color: #cbd5e0 !important;
        font-weight: 500 !important;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(45, 45, 60, 0.9);
        border-radius: 10px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        color: #cbd5e0;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(45deg, #1de9b6, #00bcd4);
        color: white !important;
    }
    
    /* Download button */
    .stDownloadButton > button {
        background: linear-gradient(45deg, #4a5568, #2d3748);
        color: #e2e8f0;
        border: 1px solid rgba(100, 255, 218, 0.3);
        border-radius: 15px;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    
    .stDownloadButton > button:hover {
        background: linear-gradient(45deg, #1de9b6, #00bcd4);
        border: 1px solid #64ffda;
        transform: translateY(-1px);
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #64ffda !important;
    }
</style>
""", unsafe_allow_html=True)

class EmotionPredictor:
    def __init__(self):
        self.model = None
        self.metadata = None
        self.is_loaded = False  # Added for deployment
        self.error_message = None  # Added for deployment
        self.load_model()
    
    def load_model(self):
        """Load the trained model and metadata"""
        try:
            # Try multiple model file options for deployment compatibility
            model_options = [
                ('models/best_single_model.pkl', 'models/best_single_metadata.pkl'),
                ('models/logistic_regression_model.pkl', 'models/best_single_metadata.pkl'),
            ]
            
            import os
            for model_file, metadata_file in model_options:
                try:
                    if os.path.exists(model_file) and os.path.exists(metadata_file):
                        # Load model
                        self.model = joblib.load(model_file)
                        
                        # Load metadata
                        with open(metadata_file, 'rb') as f:
                            self.metadata = pickle.load(f)
                        
                        self.is_loaded = True
                        st.success(f"✅ Model loaded successfully from {model_file}")
                        return True
                except Exception as e:
                    continue
            
            # If no model loaded, set error
            self.error_message = "No compatible model files found"
            return False
            
        except Exception as e:
            st.error(f"Error loading model: {e}")
            self.error_message = str(e)
            return False
    
    def clean_text(self, text):
        """Clean text for prediction"""
        if pd.isna(text) or text == "":
            return ""
        
        text = str(text).lower()
        text = re.sub(r'http\S+|@\w+|#\w+', '', text)
        text = re.sub(r'[^a-zA-Z\s!?.]', ' ', text)
        text = ' '.join(text.split())
        return text.strip()
    
    def predict_emotion(self, text):
        """Predict emotion for given text"""
        if not self.model or not self.metadata:
            return None, None, None
        
        try:
            # Clean text
            clean_text = self.clean_text(text)
            if not clean_text:
                return None, None, None
            
            # Extract features (same as training)
            tfidf_vectorizer = self.metadata['tfidf_vectorizer']
            label_encoder = self.metadata['label_encoder']
            
            # Add basic numerical features
            text_length = len(clean_text)
            word_count = len(clean_text.split())
            polarity = TextBlob(clean_text).sentiment.polarity
            
            # TF-IDF features
            tfidf_features = tfidf_vectorizer.transform([clean_text])
            
            # Combine features - with fallback for deployment
            try:
                from scipy.sparse import hstack, csr_matrix
                numerical_features = np.array([[text_length, word_count, polarity]])
                X = hstack([tfidf_features, csr_matrix(numerical_features)])
            except ImportError:
                # Fallback if scipy is not available
                tfidf_dense = tfidf_features.toarray()
                numerical_features = np.array([[text_length, word_count, polarity]])
                X = np.concatenate([tfidf_dense, numerical_features], axis=1)
            
            # Get prediction and probabilities
            prediction = self.model.predict(X)[0]
            probabilities = self.model.predict_proba(X)[0]
            
            # Convert to emotion name
            emotion_name = label_encoder.inverse_transform([prediction])[0]
            
            # Create probability dictionary
            emotion_probs = {}
            for i, emotion in enumerate(label_encoder.classes_):
                emotion_probs[emotion] = probabilities[i]
            
            # Confidence score
            confidence = max(probabilities)
            
            return emotion_name, confidence, emotion_probs
            
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return None, None, None

def create_emotion_chart(emotion_probs):
    """Create emotion probability chart"""
    if not emotion_probs:
        return None
    
    emotions = list(emotion_probs.keys())
    probabilities = [emotion_probs[emotion] * 100 for emotion in emotions]
    
    # Color mapping for emotions
    color_map = {
        'joy': '#FFD700',      # Gold
        'anger': '#FF4444',    # Red
        'sadness': '#4169E1',  # Royal Blue
        'fear': '#8A2BE2',     # Blue Violet
        'neutral': '#808080',  # Gray
        'love': '#FF69B4',     # Hot Pink
        'surprise': '#FFA500'  # Orange
    }
    
    colors = [color_map.get(emotion, '#95a5a6') for emotion in emotions]
    
    fig = go.Figure(data=[
        go.Bar(
            x=emotions,
            y=probabilities,
            marker_color=colors,
            text=[f'{p:.1f}%' for p in probabilities],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title="Emotion Probability Distribution",
        xaxis_title="Emotions",
        yaxis_title="Probability (%)",
        height=400,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    return fig

def create_emotion_pie_chart(emotion_probs):
    """Create emotion pie chart"""
    if not emotion_probs:
        return None
    
    emotions = list(emotion_probs.keys())
    probabilities = [emotion_probs[emotion] * 100 for emotion in emotions]
    
    color_map = {
        'joy': '#FFD700',
        'anger': '#FF4444', 
        'sadness': '#4169E1',
        'fear': '#8A2BE2',
        'neutral': '#808080',
        'love': '#FF69B4',
        'surprise': '#FFA500'
    }
    
    colors = [color_map.get(emotion, '#95a5a6') for emotion in emotions]
    
    fig = go.Figure(data=[
        go.Pie(
            labels=emotions,
            values=probabilities,
            marker_colors=colors,
            hole=0.3,
            textinfo='label+percent'
        )
    ])
    
    fig.update_layout(
        title="Emotion Distribution",
        height=400,
        font=dict(color='white')
    )
    
    return fig

def main():
    # Initialize predictor
    if 'predictor' not in st.session_state:
        st.session_state.predictor = EmotionPredictor()
    
    # Initialize history
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []
    
    predictor = st.session_state.predictor
    
    # Header
    st.markdown('<div class="header-container">', unsafe_allow_html=True)
    st.title("🎭 EmoSense - Emotion Detection from Text")
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <p style='font-size: 1.2rem; color: #4a5568; font-weight: 500;'>
            ✨ AI-powered emotion analysis for education, mental health, and social media ✨
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Check if model is loaded - Updated for deployment
    if not predictor.is_loaded or not predictor.model:
        st.error("❌ Model not loaded. Please ensure model files are in the 'models' directory.")
        if predictor.error_message:
            st.error(f"Error details: {predictor.error_message}")
        st.stop()
    
    # Display model info - Updated for deployment compatibility
    with st.expander("ℹ️ Model Information", expanded=False):
        if predictor.metadata:
            # Safe access to metadata fields
            model_name = predictor.metadata.get('model_name', 'Logistic Regression Model')
            accuracy = predictor.metadata.get('accuracy', 0.76)
            dataset_used = predictor.metadata.get('dataset_used', 'Emotion Dataset')
            emotion_mapping = predictor.metadata.get('emotion_mapping', {})
            
            st.write(f"**Model Type:** {model_name.replace('_', ' ').title()}")
            st.write(f"**Accuracy:** {accuracy:.1%}")
            st.write(f"**Dataset:** {dataset_used}")
            if emotion_mapping:
                st.write(f"**Emotions Detected:** {', '.join(emotion_mapping.keys())}")
            else:
                st.write("**Emotions Detected:** anger, fear, joy, sadness")
    
    # Sidebar
    st.sidebar.header("🔧 Settings")
    
    # Analysis mode
    analysis_mode = st.sidebar.selectbox(
        "Analysis Mode",
        ["Single Text", "Batch Analysis", "Real-time Chat"]
    )
    
    # Main content area
    if analysis_mode == "Single Text":
        st.header("📝 Single Text Analysis")
        
        # Text input methods
        input_method = st.radio(
            "Input Method:",
            ["Type Text", "Upload Text File"]
        )
        
        user_text = ""
        
        if input_method == "Type Text":
            user_text = st.text_area(
                "Enter text to analyze:",
                placeholder="Type your message here... (e.g., 'I'm so excited about this project!')",
                height=150
            )
        else:
            uploaded_file = st.file_uploader(
                "Upload a text file",
                type=['txt'],
                help="Upload a .txt file containing the text to analyze"
            )
            
            if uploaded_file:
                user_text = str(uploaded_file.read(), "utf-8")
                st.text_area("Uploaded text:", user_text, height=150, disabled=True)
        
        # Analysis button
        if st.button("🔍 Analyze Emotion", type="primary"):
            if user_text.strip():
                with st.spinner("Analyzing emotion..."):
                    emotion, confidence, emotion_probs = predictor.predict_emotion(user_text)
                
                if emotion:
                    # Display results in a styled card
                    st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                    
                    # Display results
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.subheader("🎯 Prediction Results")
                        
                        # Main emotion with confidence
                        confidence_color = "green" if confidence > 0.7 else "orange" if confidence > 0.5 else "red"
                        
                        # Custom styled result display
                        emotion_emoji = {
                            'joy': '😊', 'anger': '😠', 'sadness': '😢', 
                            'fear': '😰', 'neutral': '😐', 'love': '💕', 
                            'surprise': '😲'
                        }
                        
                        emoji = emotion_emoji.get(emotion, '🤔')
                        
                        st.markdown(f"""
                        <div style='text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #2d3748, #4a5568); border-radius: 15px; margin: 1rem 0; border: 1px solid rgba(100, 255, 218, 0.3);'>
                            <div style='font-size: 3rem; margin-bottom: 0.5rem;'>{emoji}</div>
                            <div style='font-size: 1.5rem; font-weight: 600; color: #64ffda; margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 2px;'>{emotion}</div>
                            <div style='font-size: 1.2rem; color: #cbd5e0;'>Confidence: {confidence:.1%}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Confidence interpretation
                        if confidence > 0.8:
                            st.success("🎉 Very Confident Prediction")
                        elif confidence > 0.6:
                            st.info("✅ Confident Prediction")
                        elif confidence > 0.4:
                            st.warning("⚠️ Moderate Confidence")
                        else:
                            st.error("❌ Low Confidence")
                        
                        # Text statistics
                        st.subheader("📊 Text Statistics")
                        
                        # Create metrics in a nice layout
                        stat_col1, stat_col2, stat_col3 = st.columns(3)
                        with stat_col1:
                            st.metric("Characters", f"{len(user_text)}")
                        with stat_col2:
                            st.metric("Words", f"{len(user_text.split())}")
                        with stat_col3:
                            sentiment_score = TextBlob(user_text).sentiment.polarity
                            st.metric("Sentiment", f"{sentiment_score:.3f}")
                    
                    with col2:
                        # Emotion probability chart
                        chart = create_emotion_chart(emotion_probs)
                        if chart:
                            st.plotly_chart(chart, use_container_width=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Detailed probabilities
                    st.subheader("📈 Detailed Emotion Probabilities")
                    prob_df = pd.DataFrame([
                        {"Emotion": emotion.title(), "Probability": f"{prob:.1%}", "Score": prob}
                        for emotion, prob in sorted(emotion_probs.items(), key=lambda x: x[1], reverse=True)
                    ])
                    
                    st.dataframe(
                        prob_df[['Emotion', 'Probability']],
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Add to history
                    st.session_state.prediction_history.append({
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'text': user_text[:100] + "..." if len(user_text) > 100 else user_text,
                        'emotion': emotion,
                        'confidence': confidence
                    })
                    
                else:
                    st.error("❌ Could not analyze the text. Please try again.")
            else:
                st.warning("⚠️ Please enter some text to analyze.")
    
    elif analysis_mode == "Batch Analysis":
        st.header("📊 Batch Analysis")
        
        uploaded_file = st.file_uploader(
            "Upload CSV file with text data",
            type=['csv'],
            help="CSV should have a 'text' column containing messages to analyze"
        )
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.write(f"📄 Loaded {len(df)} rows")
                
                # Show column selection
                text_column = st.selectbox(
                    "Select text column:",
                    df.columns.tolist()
                )
                
                if st.button("🔍 Analyze All Texts", type="primary"):
                    progress_bar = st.progress(0)
                    results = []
                    
                    for i, text in enumerate(df[text_column]):
                        emotion, confidence, _ = predictor.predict_emotion(str(text))
                        results.append({
                            'original_text': text,
                            'predicted_emotion': emotion,
                            'confidence': confidence
                        })
                        progress_bar.progress((i + 1) / len(df))
                    
                    # Display results
                    results_df = pd.DataFrame(results)
                    
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.subheader("📊 Emotion Distribution")
                        emotion_counts = results_df['predicted_emotion'].value_counts()
                        
                        fig = px.pie(
                            values=emotion_counts.values,
                            names=emotion_counts.index,
                            title="Overall Emotion Distribution"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.subheader("📈 Confidence Distribution")
                        fig = px.histogram(
                            results_df,
                            x='confidence',
                            title="Prediction Confidence Distribution",
                            nbins=20
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Results table
                    st.subheader("📋 Detailed Results")
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name=f"emotion_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
            except Exception as e:
                st.error(f"Error processing file: {e}")
    
    elif analysis_mode == "Real-time Chat":
        st.header("💬 Real-time Emotion Chat")
        
        st.info("Type messages and see emotions detected in real-time!")
        
        # Chat input
        user_input = st.chat_input("Type your message here...")
        
        if user_input:
            # Add user message
            with st.chat_message("user"):
                st.write(user_input)
            
            # Analyze emotion
            emotion, confidence, emotion_probs = predictor.predict_emotion(user_input)
            
            # Bot response
            with st.chat_message("assistant"):
                if emotion:
                    emotion_emoji = {
                        'joy': '😊', 'anger': '😠', 'sadness': '😢', 
                        'fear': '😰', 'neutral': '😐', 'love': '💕', 
                        'surprise': '😲'
                    }
                    
                    emoji = emotion_emoji.get(emotion, '🤔')
                    
                    st.write(f"{emoji} I detect **{emotion}** in your message")
                    st.write(f"Confidence: {confidence:.1%}")
                    
                    # Show top 3 emotions
                    top_emotions = sorted(emotion_probs.items(), key=lambda x: x[1], reverse=True)[:3]
                    
                    cols = st.columns(3)
                    for i, (emo, prob) in enumerate(top_emotions):
                        with cols[i]:
                            st.metric(
                                label=emo.title(),
                                value=f"{prob:.1%}"
                            )
                else:
                    st.write("😕 I couldn't analyze that message. Could you try rephrasing?")
    
    # Sidebar - Prediction History
    if st.session_state.prediction_history:
        st.sidebar.header("📜 Recent Predictions")
        
        # Show last 5 predictions
        recent_predictions = st.session_state.prediction_history[-5:]
        
        for pred in reversed(recent_predictions):
            with st.sidebar.expander(f"{pred['emotion'].title()} - {pred['confidence']:.0%}"):
                st.write(f"**Time:** {pred['timestamp']}")
                st.write(f"**Text:** {pred['text']}")
                st.write(f"**Emotion:** {pred['emotion']}")
                st.write(f"**Confidence:** {pred['confidence']:.1%}")
        
        # Clear history button
        if st.sidebar.button("🗑️ Clear History"):
            st.session_state.prediction_history = []
            st.rerun()
    
    # Footer - Updated for deployment compatibility
    st.markdown("---")
    accuracy_display = 0.76  # Default fallback
    if predictor.metadata and 'accuracy' in predictor.metadata:
        accuracy_display = predictor.metadata['accuracy']
    
    st.markdown(f"""
    <div class="footer" style="text-align: center; padding: 2rem; margin-top: 3rem;">
        <div style="background: linear-gradient(90deg, #667eea, #764ba2); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; font-size: 1.1rem; font-weight: 600;">
            🎭 EmoSense - Advanced Emotion Detection Platform
        </div>
        <div style="color: #6b7280; margin-top: 0.5rem; font-size: 0.9rem;">
            Built with ❤️ using Streamlit & Machine Learning | 
            Model Accuracy: {accuracy_display:.1%}
        </div>
        <div style="color: #9ca3af; margin-top: 0.25rem; font-size: 0.8rem;">
            Empowering emotional intelligence through AI
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
