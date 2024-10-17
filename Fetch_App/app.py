import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
from predict import generate_prediction
from train import train_model

# Function that returns all available checkpoints
def list_checkpoints(folder='checkpoints'):
    if not os.path.exists(folder):
        return []
    files = [f for f in os.listdir(folder) if f.endswith('.pth')]
    return files

# Function to plot daily or monthly data
def plot_data(data, view_type, year):
    data['Date'] = pd.to_datetime(data['# Date'])
    if view_type == 'Daily':
        plt.figure(figsize=(10, 5))
        plt.plot(data['Date'], data['Receipt_Count'], color='blue', label="Daily Data")
        plt.xlabel('Date')
        plt.ylabel('Receipt Count')
        plt.title(f'Observed Receipts in {year}- Daily')
        st.pyplot(plt)
    elif view_type == 'Monthly':
        data['Month'] = data['Date'].dt.to_period('M')
        monthly_data = data.groupby('Month')['Receipt_Count'].sum().reset_index()
        plt.figure(figsize=(10, 5))
        plt.bar(monthly_data['Month'].astype(str), monthly_data['Receipt_Count'], color='orange')
        plt.xlabel('Month')
        plt.ylabel('Total Receipt Count')
        plt.title(f'Observed Receipts in {year} - Monthly Sum')
        st.pyplot(plt)

def main():
    st.title('Receipt Count Prediction App')
    # Initialize rerun counter in session_state if it doesn't exist
    if 'rerun_counter' not in st.session_state:
        st.session_state['rerun_counter'] = 0
    data_path = st.file_uploader('Upload the previous year data (CSV)', type='csv')
    if data_path:
        data = pd.read_csv(data_path)
        st.session_state['data'] = data
    if 'data' in st.session_state:
        data = st.session_state['data']
        # Display toggle
        view_type = st.radio('Select view:', ['Daily', 'Monthly'], index=1)
        # Model selection
        st.sidebar.subheader('Model Configuration')
        model_choice = st.sidebar.selectbox(
            'Select Model to Use:',
            ['LSTM', 'MLP'],
            index=1
        )
        hidden_size = st.sidebar.slider('Hidden Size', min_value=10, max_value=100, value=30)
        epochs = st.sidebar.slider('Epochs', min_value=10, max_value=100, value=50)
        year = st.sidebar.selectbox('Prediction Year', [2022])
        st.write(f'Original Data ({year - 1}):')
        plot_data(data, view_type, year - 1)
        st.sidebar.subheader('Checkpoint Management')
        # Checkpoint selection
        checkpoint_files = list_checkpoints()
        if checkpoint_files:
            checkpoint_choice = st.sidebar.selectbox('Select a checkpoint to use for prediction:', checkpoint_files)
            checkpoint_path = os.path.join('checkpoints', checkpoint_choice)
            st.session_state['checkpoint_path'] = checkpoint_path
        else:
            st.sidebar.write("No checkpoints found. Please train a model first.")

        # Train the model when the button is clicked
        if st.sidebar.button('Train Model'):
            st.sidebar.write(f'Training model: {model_choice}...')
            # Save the uploaded CSV file for use in training
            temp_data_path = os.path.join('data', 'temp_data.csv')
            data.to_csv(temp_data_path, index=False)
            # Call the train function from train.py with the chosen model
            args_train = argparse.Namespace(
                data_path=temp_data_path,
                model=model_choice,  
                hidden_size=hidden_size,
                epochs=epochs
            )
            checkpoint_path = train_model(args_train)  
            st.session_state['checkpoint_path'] = checkpoint_path
            checkpoint_files = list_checkpoints()  
            # Increment rerun counter to trigger update
            st.session_state['rerun_counter'] += 1

        # Prediction generation
        if st.sidebar.button('Predict Next year'):
            if 'checkpoint_path' in st.session_state:
                st.sidebar.write('Generating predictions...')
                temp_data_path = os.path.join('data', 'temp_data.csv')
                args_predict = argparse.Namespace(data_path=temp_data_path, checkpoint=st.session_state['checkpoint_path'], year=year)
                predictions_df, file_name = generate_prediction(args_predict)
                st.session_state['predictions_df'] = predictions_df
                st.session_state['file_name'] = file_name
                st.sidebar.write('Predictions Complete.')
                st.session_state['rerun_counter'] += 1
            else:
                st.sidebar.write("Please train the model or select a checkpoint.")

        if 'predictions_df' in st.session_state:
            st.subheader(f"Predictions for {year}")
            view_type = st.radio('Select view for predictions:', ['Daily', 'Monthly'], key='view_predictions', index=1)
            plot_data(st.session_state['predictions_df'], view_type, year)

            # Downloader
            csv = st.session_state['predictions_df'].to_csv(index=False)
            st.download_button(
                label="Download Predictions as CSV",
                data=csv,
                file_name=st.session_state['file_name'],
                mime='text/csv',
            )

if __name__ == '__main__':
    main()
