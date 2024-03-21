import os
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy import signal
from scipy.signal import butter, freqz, filtfilt

def preprocess_data(data):
    # Normalization: Standardize the data
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(data)

    # Cleaning: Remove any rows with missing values or outliers
    cleaned_data = data.dropna()  # Drop rows with missing values

    # Feature Engineering: Extract relevant features (optional, depending on the model)
    # Example: Calculate magnitude from accelerometer, gyroscope, gravity, and attitude readings
    cleaned_data['acceleration_magnitude'] = np.sqrt(cleaned_data['userAcceleration.x'] ** 2 +
                                                     cleaned_data['userAcceleration.y'] ** 2 +
                                                     cleaned_data['userAcceleration.z'] ** 2)
    cleaned_data['rotation_magnitude'] = np.sqrt(cleaned_data['rotationRate.x'] ** 2 +
                                                 cleaned_data['rotationRate.y'] ** 2 +
                                                 cleaned_data['rotationRate.z'] ** 2)
    cleaned_data['attitude_magnitude'] = np.sqrt(cleaned_data['attitude.roll'] ** 2 +
                                                 cleaned_data['attitude.pitch'] ** 2 +
                                                 cleaned_data['attitude.yaw'] ** 2)

    # Keep only the magnitude features
    magnitude_columns = ['acceleration_magnitude', 'rotation_magnitude', 'attitude_magnitude']
    cleaned_data = cleaned_data[magnitude_columns]

    return cleaned_data


# Function to determine the number of chunks for a given prefix
def determine_num_chunks(prefix):
    i = 0
    while True:
        # Construct the file name
        file_name = prefix.format(i)
        # Check if this file name exists
        if os.path.isfile(file_name):
            i += 1
        else:
            break
    return i


def calculate_cosine_similarity(chunk_A, chunk_B):
    # Calculate cosine similarity between two chunks
    similarity_scores = cosine_similarity(chunk_A, chunk_B)
    return similarity_scores


def categorize_recovery_by_percent(similarity_score):
    if similarity_score <= 0.25:
        return "0% to 25% recovery"
    elif similarity_score <= 0.5:
        return "25% to 50% recovery"
    elif similarity_score <= 0.75:
        return "50% to 75% recovery"
    else:
        return "75% to 100% recovery"


def categorize_recovery_by_stage(similarity_score):
    if similarity_score <= 0.25:
        return "Early recovery"
    elif similarity_score <= 0.5:
        return "Moderate recovery"
    elif similarity_score <= 0.75:
        return "Advanced recovery"
    else:
        return "Full recovery"


def lowpass_filter(data, low_cut_off=1, fs=50):
    b, a = butter(4, low_cut_off, fs=fs, btype='lowpass', analog=False)
    y = filtfilt(b, a, data)
    return b, a, y
    

def calculate_user_acceleration(df):
    # Create a new dataframe to store the acceleration data
    df_acceleration = pd.DataFrame()
    
    # Copy columns 'acceleration.x', 'acceleration.y', 'acceleration.z' from the original dataframe into df_acceleration
    df_acceleration['x'] = df['acceleration.x']
    df_acceleration['y'] = df['acceleration.y']
    df_acceleration['z'] = df['acceleration.z']
    
    # Obtain gravity component for 'x' axis
    b, a, df_acceleration['x_g'] = lowpass_filter(df_acceleration['x'].values, low_cut_off=1, fs=50)

    # Obtain gravity component for 'y' axis
    b, a, df_acceleration['y_g'] = lowpass_filter(df_acceleration['y'].values, low_cut_off=1, fs=50)

    # Obtain gravity component for 'z' axis
    b, a, df_acceleration['z_g'] = lowpass_filter(df_acceleration['z'].values, low_cut_off=1, fs=50)
    
    # Calculate userAcceleration for 'y' axis
    df_acceleration['userAcceleration.y'] = df_acceleration['y'] - df_acceleration['y_g']
    
    # Calculate userAcceleration for 'z' axis
    df_acceleration['userAcceleration.z'] = df_acceleration['z'] - df_acceleration['z_g']
    
    # Calculate userAcceleration for 'x' axis
    df_acceleration['userAcceleration.x'] = df_acceleration['x'] - df_acceleration['x_g']
    
    # Replace columns 'acceleration.x', 'acceleration.y', 'acceleration.z' in the original dataframe with
    # the userAcceleration.y, userAcceleration.z, userAcceleration.x columns of df_acceleration
    df[['acceleration.x', 'acceleration.y', 'acceleration.z']] = df_acceleration[['userAcceleration.x', 'userAcceleration.y', 'userAcceleration.z']]
    
    # Return the updated dataframe
    return df
