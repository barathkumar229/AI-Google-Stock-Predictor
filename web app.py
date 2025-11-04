# -*- coding: utf-8 -*-
import yfinance as yf
import numpy as np
import pandas as pd
import pickle
import streamlit as st
from datetime import date, timedelta
import time
from streamlit_lottie import st_lottie
import json


# --- CONFIGURATION & STYLING ---
# Set the page title, icon, and wide layout
import base64

def add_bg_from_local(image_file):
    """Adds a background image from a local file to Streamlit."""
    # NOTE: In a cloud environment, you must use a public URL instead of a local path.
    # The image path is local and likely won't resolve outside the user's machine.
    # Keeping the structure for local testing flexibility.
    try:
        with open(image_file, "rb") as f:
            encoded_string = base64.b64encode(f.read()).decode()
        st.markdown(
            f"""
            <style>
            [data-testid="stAppViewContainer"] {{
                background-image: url("data:image/png;base64,{encoded_string}");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                background-attachment: fixed;
                /* Forcefully disable vertical scroll on the main container */
                overflow-y: hidden !important; 
            }}
            [data-testid="stHeader"] {{
                background: rgba(0,0,0,0);
            }}
            [data-testid="stToolbar"] {{
                right: 2rem;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    except Exception:
        # Fallback if image path is invalid in the execution environment
        st.warning("⚠️ Background image path could not be loaded. Check local file path resolution.")


# ✅ Call it right after st.set_page_config
add_bg_from_local(r"C:\Users\Barathkumar\Downloads\jamie-street-VP4WmibxvcY-unsplash.jpg")

st.set_page_config(
    page_title="Fast Stock Predictor", 
    layout="wide", 
    initial_sidebar_state="collapsed", 
    menu_items={'About': "This app uses an LSTM model to predict Google's stock price."}
)

# Custom CSS for a cleaner, more modern look
st.markdown("""
<style>
/* --- FIX: Forcefully disable vertical scrolling (Scroll Block) --- */
/* Target the overall HTML and body */
html, body {
    overflow-y: hidden !important;
}

/* --- FIX: Aggressively reduce Streamlit's default vertical padding to eliminate extra scrolling --- */
[data-testid="stAppViewContainer"] > .main > .block-container {
    padding-top: 0rem !important; /* Reduced to 0 for maximum compactness */
    padding-bottom: 0rem !important; /* Reduced to 0 for maximum compactness */
    overflow-y: hidden !important; /* Block scroll on the main content container */
}

/* Target all containers that might hold scrollable content, including Streamlit's specific wrappers */
.main > div {
    overflow-y: hidden !important;
}
[data-testid="stVerticalBlock"], 
[data-testid="stVerticalBlockBorderWrapper"],
[data-testid="stBlock"],
[data-testid="stHorizontalBlock"] {
    overflow-y: hidden !important;
    max-height: 100vh !important; /* Set maximum height to viewport height */
}
/* ----------------------------------------------------------------------------------- */

/* Adjust main app background and font */
.main {
    background-color: #f0f2f6; /* Fallback color */
    /* Add background image properties - This is redundant due to the Python function above, but kept for clarity on original intent */
    background-image: url("C:/Users\Barathkumar/Downloads/pexels-alesiakozik-6772076.jpg"); 
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed; /* Ensures content scrolls over a static background */
}
/* Style the main title */
h1 {
    color: #004c99; /* Deep corporate blue for the main title */
    text-align: left; /* Aligned left, near the margin */
    font-weight: 900; /* Increased to 900 for extra professionalism */
    font-size: 3rem; /* Increased size for stronger presence */
    padding-top: 0px; /* Set to 0px to move closer to the top margin */
    padding-bottom: 15px; /* Reduced from 25px to save space */
    margin-top: -10px; /* Pulls the heading up slightly more for a tight top margin */
}
/* Style the subheaders */
h3, h4 {
    color: #343a40; /* Dark gray for headings */
}

/* === ENHANCED PREDICT BUTTON STYLE === */
div.stButton > button {
    display: block;
    margin: 20px auto; /* Increased margin */
    width: 90%; /* Slightly wider */
    padding: 12px 20px;
    font-size: 1.3rem;
    font-weight: 600;
    /* Deep Corporate Blue with subtle gradient */
    background-color: #004c99;
    background-image: linear-gradient(180deg, #0066cc 0%, #004c99 100%);
    color: white;
    border-radius: 10px;
    border: none;
    box-shadow: 0 4px 10px rgba(0, 76, 153, 0.3); /* Blue shadow */
    transition: all 0.3s ease;
    letter-spacing: 0.5px;
    cursor: pointer;
}
div.stButton > button:hover {
    background-color: #007bff;
    background-image: none;
    box-shadow: 0 6px 15px rgba(0, 76, 153, 0.5); /* Stronger shadow on hover */
    transform: translateY(-2px); /* Slight lift effect */
}


/* Style the containers/boxes */
/* Added a slight transparent white background to the general container for contrast against the image */
.stContainer {
    border: 1px solid #dee2e6;
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 20px;
    background-color: rgba(255, 255, 255, 0.95); /* Semi-transparent white */
    box-shadow: 0 4px 8px rgba(0,0,0,0.05);
    /* === FULLY DISABLE SCROLLING IN STREAMLIT APP === */

/* Disable vertical scrolling globally */
html, body {
    overflow: hidden !important;
    height: 100vh !important;
}

/* Prevent Streamlit’s layout containers from creating internal scrollbars */
[data-testid="stAppViewContainer"],
[data-testid="stVerticalBlock"],
[data-testid="stVerticalBlockBorderWrapper"],
[data-testid="stBlock"],
[data-testid="stHorizontalBlock"],
.main, .block-container {
    overflow: hidden !important;
    height: 100vh !important;
}

/* Ensure Streamlit columns and frames don’t push overflow */
div[data-testid="column"], div[data-testid="stHorizontalBlock"] > div {
    overflow: hidden !important;
}

/* Remove padding that can create scroll bars */
.block-container {
    padding-top: 0 !important;
    padding-bottom: 0 !important;
}

/* Optional: Hide scrollbars visually too */
::-webkit-scrollbar {
    display: none;
}

/* === END SCROLL DISABLE SECTION === */
}

</style>
""", unsafe_allow_html=True)


# --- LOTTIE UTILITY ---

@st.cache_data
def load_lottie_file(filepath: str):
    """Safely loads a Lottie animation file."""
    try:
        # NOTE: In a cloud environment, you must use a public URL instead of a local path.
        # Assuming the path is relative or managed correctly in the execution environment.
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        # Fallback if file isn't accessible (e.g., local path fails in web environment)
        st.warning("Could not load local Lottie file. Using a placeholder.")
        return None # In a true environment, this should be replaced with a public URL fetch.

# ✅ Load animation safely (Update this path)
# WARNING: This path is local and will likely fail in a cloud environment.
lottie_animation = load_lottie_file(
    "C:/Users/Barathkumar/OneDrive/Desktop/livegoogle stock prediction/Fast Stock Market App.json"
)


# --- MODEL LOADING ---

@st.cache_resource
def load_model(filepath: str):
    """Safely loads the pickled model."""
    try:
        return pickle.load(open(filepath, 'rb'))
    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
        return None

# ✅ Load your model (Update this path)
# WARNING: This path is local and will likely fail in a cloud environment.
loaded_model = load_model("C:/Users/Barathkumar/Downloads/trained_model.sav")


# --- DATA FETCHING & PREDICTION LOGIC (Kept for robustness) ---

def live_stock_data():
    """Fetches GOOGL data, preprocesses it, and returns the prediction or an error string."""
    if loaded_model is None:
        return 'Model Load Error: The prediction model failed to load at startup.'
        
    yesterday = date.today() - timedelta(days=1)
    # Fetch enough data to calculate the 7-day moving average (MA7)
    start_date = yesterday - timedelta(days=30) 

    try:
        data = yf.download("GOOGL", start=start_date, end=yesterday, progress=False)

        if data.empty:
            return 'Data Fetch Error: Could not download recent data for GOOGL.'

        data.reset_index(inplace=True)
        data = data.rename(columns={'Adj Close': 'Close'}) 

        # Feature Engineering 
        data['Price Changed'] = data['Close'] - data['Open']
        data['MA7'] = data['Close'].rolling(7).mean()

        data = data.dropna()
        if data.empty:
            return 'Data Processing Error: Not enough data points after calculating MA7.'
            
        latest_data = data.tail(1).copy()
        
        feature_columns = ['High', 'Low', 'Open', 'Volume', 'Price Changed', 'MA7']
        # Reshape for prediction (assuming the model expects shape (1, n_features))
        X = latest_data[feature_columns].values.reshape(1, -1) 
        
        y_pred = loaded_model.predict(X)
        
        # Ensure output is a standard float for display
        return float(y_pred[0]) 
        
    except Exception as e:
        return f'Prediction Runtime Error: {e}.'


# --- STREAMLIT APP CORE ---

def main_app():
    # --- 1. INITIAL LOADING SCREEN ---
    if 'app_loaded' not in st.session_state:
        st.session_state.app_loaded = False
        
    if not st.session_state.app_loaded:
        loading_placeholder = st.empty()

        with loading_placeholder.container():
            st.markdown(
                "<h3 style='text-align:center;'>🚀 Initializing Model and Loading Resources...</h3>",
                unsafe_allow_html=True,
            )
            if lottie_animation:
                st_lottie(lottie_animation, speed=1, loop=True, height=250, key="initial_loading_screen")
            
        time.sleep(1) 
        
        loading_placeholder.empty()
        st.session_state.app_loaded = True
        st.rerun() 

    # --- 2. MAIN APP UI ---
    
    # Use columns to center content and add side margins
    col1, col_center, col3 = st.columns([1, 4, 1])
    
    # Placeholders to control visibility in the center column
    main_ui_placeholder = col_center.empty()
    
    # Initialize session state
    if 'is_predicting' not in st.session_state:
        st.session_state.is_predicting = False
    if 'prediction_result_msg' not in st.session_state:
        st.session_state.prediction_result_msg = None

    # A. Display Main UI (If not currently predicting)
    if not st.session_state.is_predicting:
            
        with main_ui_placeholder.container():
            st.title("📈 Fast Stock Market App")
            
            # Use a separate container for the main action area
            with st.container(border=True):
                st.markdown("<h4 style='text-align:center;'>Google Stock Prediction Dashboard</h4>", unsafe_allow_html=True)
                st.info('This tool uses a trained model to predict the next closing price for **GOOGL** based on historical data.', icon="💡")
                
                # Check for and display prediction result here, inside the main container
                if st.session_state.prediction_result_msg is not None:
                    st.markdown(st.session_state.prediction_result_msg, unsafe_allow_html=True)
                    # Clear the message after displaying it
                    st.session_state.prediction_result_msg = None 
                
                # Center the button using Markdown/CSS 
                if st.button('Predict Next Close Price'):
                    st.session_state.is_predicting = True
                    st.rerun() 
                    
            st.markdown("---")
            st.markdown("<p style='text-align:center; color:#6c757d;'>Data powered by Yahoo! Finance. Model: LSTM.</p>", unsafe_allow_html=True)

    # B. Handle Prediction Process (If is_predicting is True)
    if st.session_state.is_predicting:
        
        # 1. Hide the main UI container
        main_ui_placeholder.empty()
        
        # 2. Display the prediction loading screen in the center column
        prediction_loading_placeholder = col_center.empty()

        # --- MODIFICATION: Removed border=True from st.container() ---
        with prediction_loading_placeholder.container(): 
            # Note: Removed color:#ffc107 style to keep it more subdued
            st.markdown(
                "<h3 style='text-align:center;'>🔮 Fetching Live Data and Predicting...</h3>",
                unsafe_allow_html=True,
            )
            if lottie_animation:
                st_lottie(lottie_animation, speed=1, loop=True, height=200, key="prediction_loading")
            
            with st.spinner('Waiting for data and running model...'):
                time.sleep(1) 
                y_pred = live_stock_data()

        # 3. Process and store the result
        
        if isinstance(y_pred, float):
            predicted_value = y_pred
            # Store simpler styled success message for direct insertion into the main UI
            message = f"""
<div style='background-color: #e6f7ff; border: 2px solid #007bff; border-radius: 8px; padding: 15px; margin-bottom: 20px; text-align: center;'>
    <h5 style='color:#004c99; margin: 0; font-weight: 600;'>PREDICTED CLOSING PRICE FOR GOOGL</h5>
    <div style='padding: 10px 0;'>
        <strong style='color:#004c99; font-size: 2.5rem; font-weight: 800;'>${predicted_value:.2f}</strong>
    </div>
    <p style='color: #28a745; margin: 0; font-size: 0.9rem;'>
        ✅ Prediction successful. Click the button below to predict again.
    </p>
</div>
            """
            st.session_state.prediction_result_msg = message
        elif isinstance(y_pred, str):
            error_msg = y_pred
            # Store styled error message for direct insertion into the main UI
            message = f"""
<div style='background-color: #fff0f0; border: 2px solid #dc3545; border-radius: 8px; padding: 15px; margin-bottom: 20px; text-align: left;'>
    <h5 style='color:#dc3545; margin: 0; font-weight: 600;'>❌ PREDICTION FAILED</h5>
    <p style='font-size: 1rem; margin-top: 10px; color: #343a40;'>
        The prediction could not be completed due to a runtime issue.
    </p>
    <div style='background-color: #f7f9fc; padding: 10px; border-radius: 4px; font-size: 0.9rem;'>
        <strong>Detail:</strong> <code>{error_msg}</code>
    </div>
</div>
            """
            st.session_state.prediction_result_msg = message
        else:
            st.session_state.prediction_result_msg = "<h4 style='color:red;'>❌ Prediction Failed!</h4>Reason: Unexpected output from model function."
            
        # 4. Reset state and rerun to show the result
        st.session_state.is_predicting = False
        prediction_loading_placeholder.empty() 
        st.rerun() 


if __name__ == '__main__':
    main_app()
