import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib   
import matplotlib.pyplot as plt

model = joblib.load('/mount/src/step2_gui/xgb_multioutput_model_GUI_compressed.pkl')
scaler = joblib.load('/mount/src/step2_gui/scaler.pkl')  # Load the scaler used during training
st.title("XGBoost Model Prediction GUI")
st.write("Enter the design parameters to get the predicted S21 values.")
# Define input fields for design parameters
HeatSprd_thick = st.number_input("HeatSprd_thick (in mm)", min_value=0.0, max_value=20.0, value=2.5, step=0.1)
port3_yloc = st.number_input("port3_yloc (in mm)", min_value=-96.0, max_value=100.0, value=0.0, step=0.5)
port_depth = st.number_input("port_depth (in mm)", min_value=0.0, max_value=140.0, value=49.0, step=0.1)
Stick_W = st.number_input("Stick_W (in mm)", min_value=0.0, max_value=100.0, value=32.0, step=0.1)
Stick_L = st.number_input("Stick_L (in mm)", min_value=0.0, max_value=300.0, value=35.0, step=0.5)  
# Create a DataFrame for the input parameters and scale them
input_data = pd.DataFrame({
    'HeatSprd_thick': [HeatSprd_thick],
    'port3_yloc': [port3_yloc],
    'port_depth': [port_depth],
    'Stick_W': [Stick_W],
    'Stick_L': [Stick_L]
})

#Displaying a CAD model picture
st.subheader("CAD Model (Fire TV Stick Abstraction) used for training")
st.image("/mount/src/step2_gui/CAD_model.png", caption="CAD Model", width=True)

input_data_scaled = scaler.transform(input_data)
st.write("Scaled Input Data:")
st.write(scaler.transform([input_data.iloc[0]]))  # Display the scaled input data]))

# Note: Ensure that the input parameters are within the ranges used during model training for accurate predictions.
# Display a note about input parameter ranges
st.markdown("**Note:** The input parameters are within the following ranges for training data:")
st.markdown("- HeatSprd_thick: 0 to 5 mm")
st.markdown("- port3_yloc: -48 to 48 mm")  
st.markdown("- port_depth: 30 to 68 mm")
st.markdown("- Stick_W: 12.8 to 51.2 mm")
st.markdown("- Stick_L: 35 to 140 mm")

#verify the input data
st.subheader("Current Input Design Parameters")
st.write(input_data)
if 'counter' not in st.session_state:
    st.session_state.counter = 0
#predict button
if st.button("Predict S21"):
    st.session_state.counter += 1
    prediction = pd.DataFrame({'S21_dB': model.predict(input_data_scaled)[0]})

    st.subheader("Predicted S21 Values (in dB)")
    freq = np.linspace(2, 3, 1001)
    fig, ax = plt.subplots()
    ax.plot(freq, prediction.iloc[:, 0].astype(np.float32), label='Predicted S21', color='blue')
    ax.set_title('Predicted S21 vs Frequency')
    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel('S21 (dB)')
    ax.legend()
    ax.grid()
    st.pyplot(fig)

    # --- Manage prediction and input history ---
    if 'total_predictions' not in st.session_state:
        st.session_state.total_predictions = pd.DataFrame()
    if 'input_history' not in st.session_state:
        st.session_state.input_history = pd.DataFrame()

    # Append prediction
    st.session_state.total_predictions[f"Prediction {st.session_state.counter}"] = prediction.iloc[:, 0].astype(np.float32)
    # Append input parameters
    input_row = input_data.copy()
    input_row['Prediction'] = f"Prediction {st.session_state.counter}"
    st.session_state.input_history = pd.concat([st.session_state.input_history, input_row], ignore_index=True)

    # Reset if more than 5 predictions
    if st.session_state.counter > 5:
        st.session_state.counter = 1
        st.session_state.total_predictions = pd.DataFrame()
        st.session_state.input_history = pd.DataFrame()
        st.warning("Maximum of 5 predictions reached. Resetting counter.")

    st.success("Prediction completed!")


# Ensure session state variables are initialized before use
if 'counter' not in st.session_state:
    st.session_state.counter = 0
if 'total_predictions' not in st.session_state:
    st.session_state.total_predictions = pd.DataFrame()
if 'input_history' not in st.session_state:
    st.session_state.input_history = pd.DataFrame()



# Show predictions and input history
st.write("Total Predictions so far:")
st.dataframe(st.session_state.total_predictions)
st.write("Input Design Parameters for Predictions:")
st.dataframe(st.session_state.input_history)

#adding multiple prediction outputs to the existing plot
#predictions to be added sepetated by comma and can only be added a max of 5 plots
#adding columns from session_state.total_predictions to the existing plot as per user input using the prompt
st.write("Counter: ",st.session_state.counter)
pred_cols = st.text_input("Enter a list of prediction numbers to add (1-5) separated by commas", key="pred_input")

if st.button("Add Predictions to Plot"):
    st.write("You entered: ", pred_cols)
    if 'total_predictions' in st.session_state and not st.session_state.total_predictions.empty:
        try:
            pred_list = [int(x.strip()) for x in str(pred_cols).split(',') if x.strip().isdigit() and 1 <= int(x.strip()) <= 5]
            if not pred_list:
                st.error("Please enter valid prediction numbers between 1 and 5.")
            else:
                #clear the existing plot and re-plot all selected predictions
                #ax.clear()
                fig, ax = plt.subplots()
                freq = np.linspace(2, 3, 1001)  # Frequency range from 2 GHz to 3 GHz
                for pred_num in pred_list:
                    col_name = f"Prediction {pred_num}"
                    if col_name in st.session_state.total_predictions.columns:
                        ax.plot(freq, st.session_state.total_predictions[col_name], label=col_name)
                    else:
                        st.warning(f"{col_name} does not exist.")
                
                ax.set_title('Predicted S21 vs Frequency')
                ax.set_xlabel('Frequency (GHz)')
                ax.set_ylabel('S21 (dB)')
                ax.legend()
                ax.grid()
                st.pyplot(fig)
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.error("No predictions available to add. Please make predictions first.")


