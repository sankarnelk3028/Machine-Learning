import os
import numpy as np
import pickle
import streamlit as st

# Load the model with error handling
model_path = '/home/sankar/Downloads/Project /Final/Performance.pkl'
try:
    with open(model_path, 'rb') as file:
        Performance = pickle.load(file)
except FileNotFoundError:
    st.error(f"Model file not found at path: {model_path}. Please check the path to the model file.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")
    st.stop()

# Streamlit application title and cricket-themed images
st.title("🏏 ODI Cricket Innings Performance Prediction Model")
st.subheader("Based on 20th Century Stats\nBy: Sankar Krishnan")

st.subheader("""
    Description of the Model
    
    This model utilizes a combination of key batting and bowling statistics to predict
    the overall performance of a cricket player in an ODI match. The model is built on 
    historical data from 20th-century ODI players and applies machine learning techniques 
    to identify patterns in how these stats correlate with success in matches.
""")

# Sidebar inputs with cricket icons
st.sidebar.image("https://t4.ftcdn.net/jpg/06/15/06/81/360_F_615068142_3siXSo1TWv7ebrf7OR0XdHurjzkcpJE3.jpg", width=200)
st.sidebar.title("Cricket Player Stats Input")

# Input fields with clear labels and explanations
Innings_Runs_Scored = st.sidebar.slider("Runs Scored", min_value=0, max_value=300, value=0, step=1)
Innings_Balls_Faced = st.sidebar.slider("Total Balls Faced", min_value=Innings_Runs_Scored // 6, max_value=300, value=Innings_Runs_Scored // 6, step=1)
Number_of_overs_bowled = st.sidebar.selectbox("Total Overs Completely Bowled", list(range(11)))  # Include 10 in the range

# Initialize Balls_bowled_last_over to 0
Balls_bowled_last_over = 0

# Conditional input for Balls Bowled Last Over
if Number_of_overs_bowled < 10:
    Balls_bowled_last_over = st.sidebar.selectbox("Balls Bowled Last Over", [0, 1, 2, 3, 4, 5])
else:
    Balls_bowled_last_over = 0

# Calculate Innings_Overs_Bowled
Innings_Overs_Bowled = Number_of_overs_bowled + (Balls_bowled_last_over / 10)

# Ensure max values for sliders are valid
if Innings_Runs_Scored > 0:
    max_boundary_fours = max(Innings_Runs_Scored // 4, 1) 
    Innings_Boundary_Fours = st.sidebar.slider(
        "Boundary Fours", min_value=0, max_value=max_boundary_fours, value=0, step=1
    )

    max_boundary_sixes = max((Innings_Runs_Scored - (Innings_Boundary_Fours * 4)) // 6, 0)
    Innings_Boundary_Sixes = st.sidebar.slider(
        "Boundary Sixes", min_value=0, max_value=max_boundary_sixes, value=0, step=1
    )
else:
    # If no runs are scored, both fours and sixes sliders should default to 0
    Innings_Boundary_Fours = 0
    Innings_Boundary_Sixes = 0

if Innings_Overs_Bowled > 0:
    Innings_Wickets_Taken = st.sidebar.slider("Wickets Taken", min_value=0, max_value=10, value=0)
    Innings_Maidens_Bowled = st.sidebar.slider("Maidens Bowled", min_value=0, max_value=Number_of_overs_bowled, value=0, step=1)
    Innings_Runs_Conceded = st.sidebar.slider("Runs Conceded", min_value=0, max_value=150, value=0, step=1)
else:
    Innings_Wickets_Taken = 0
    Innings_Maidens_Bowled = 0
    Innings_Runs_Conceded = 0

# Display inputs with clear labels and explanations
st.write("Innings Balls Faced:", Innings_Balls_Faced)
st.write("Innings Runs Scored:", Innings_Runs_Scored)
st.write("Innings Boundary Fours:", Innings_Boundary_Fours)
st.write("Innings Boundary Sixes:", Innings_Boundary_Sixes)
st.write('Innings Overs Bowled:', Innings_Overs_Bowled)
st.write("Innings Wickets Taken:", Innings_Wickets_Taken)
st.write("Innings Maidens Bowled:", Innings_Maidens_Bowled)
st.write("Innings Runs Conceded:", Innings_Runs_Conceded)

# Predict button with cricket emoji
if st.button("Predict 🏏"):
    x = np.array([[Innings_Runs_Scored, Innings_Balls_Faced, Innings_Boundary_Fours,
                   Innings_Boundary_Sixes, Innings_Wickets_Taken, Innings_Overs_Bowled,
                   Innings_Maidens_Bowled, Innings_Runs_Conceded]])

    # Add dummy features to match the expected number of features
    num_features_expected = 28
    num_features_current = x.shape[1]
    if num_features_current < num_features_expected:
        x = np.hstack([x, np.zeros((x.shape[0], num_features_expected - num_features_current))])

    # Load the model with error handling
    model_path = '/home/sankar/Downloads/Project /Final/Performance.pkl'
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
    except FileNotFoundError:
        st.error(f"Model file not found at path: {model_path}. Please check the path to the model file.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        st.stop()

    # Make prediction
    try:
        y_predict = model.predict(x)
        st.success(f'Prediction is {y_predict[0]} 🏆')
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
