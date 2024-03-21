from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from functions import preprocess_data, calculate_cosine_similarity, categorize_recovery_by_percent, categorize_recovery_by_stage, determine_num_chunks, lowpass_filter, calculate_user_acceleration

app = Flask(__name__)

# Analyze data route
@app.route('/analyze-data', methods=['POST'])
def analyze_data():
    try:
    
        # Receive JSON data from the POST request
        json_data = request.json

        # Convert JSON data to DataFrame
        df = pd.json_normalize(json_data)

        # Define columns to remove
        columns_to_remove = ['index', 'gravity.x', 'gravity.y', 'gravity.z']

        # Remove specified columns if they exist
        for column in columns_to_remove:
            if column in df.columns:
                df.drop(columns=[column], inplace=True)
                
         
        # Get the first 45000 rows from the df
        if len(df) >= 45000:
            return df.head(45000)
        # else:
            # Return a 400 error if the dataframe has less than 45000 rows
            # return flask.jsonify({'error': 'Dataframe does not have at least 45000 rows'}), 400
            
          
        # Function to calculate user acceleration
        df = calculate_user_acceleration(df)
            

        sensor_columns = ['userAcceleration.x', 'userAcceleration.y', 'userAcceleration.z',
                          'rotationRate.x', 'rotationRate.y', 'rotationRate.z',
                          'attitude.roll', 'attitude.pitch', 'attitude.yaw']

        # Preprocess the data
        preprocessed_data = preprocess_data(df[sensor_columns])

        # Initialize MinMaxScaler
        scaler = MinMaxScaler()

        # Normalize the features
        normalized_data = pd.DataFrame(scaler.fit_transform(preprocessed_data), columns=preprocessed_data.columns)

        # Define the chunk size
        chunk_size = 10000  # Adjust this value as needed based on your available memory

        # No need to split the dataframe, work with the entire dataset
        df_B = normalized_data

        # file_path_A & file_path_B are defined
        file_path_A = 'df_A_chunk_{}.csv'
        file_path_B = 'df_B_chunk_{}.csv'

        # Write DataFrame chunks to files
        for i, chunk in enumerate(df_B.groupby(np.arange(len(df_B)) // chunk_size)):
            chunk[1].to_csv(file_path_B.format(i), index=False)

        num_chunks_A = determine_num_chunks(file_path_A)
        num_chunks_B = determine_num_chunks(file_path_B)

        # Initialize variables to store total similarity score and count of similarity scores
        total_similarity_score = 0
        num_similarity_scores = 0

        # Loop over the number of chunks in A
        for i in range(num_chunks_A):
            chunk_A = pd.read_csv(file_path_A.format(i))

            # Check if the corresponding chunk for B exists
            if i < num_chunks_B:
                chunk_B = pd.read_csv(file_path_B.format(i))
                similarity_scores = calculate_cosine_similarity(chunk_A, chunk_B)

                # Calculate the total similarity score and count of similarity scores
                total_similarity_score += np.sum(similarity_scores)
                num_similarity_scores += np.size(similarity_scores)

        # Calculate the average similarity score
        average_similarity_score = total_similarity_score / num_similarity_scores if num_similarity_scores > 0 else 0

        # Determine recovery categories based on similarity score
        recovery_category_by_percent = categorize_recovery_by_percent(average_similarity_score)
        recovery_category_by_stage = categorize_recovery_by_stage(average_similarity_score)

        # Convert average similarity score to percentage
        actual_similarity_percentage = average_similarity_score * 100

        # Return the evaluation results
        evaluation_results = {
            'actual_similarity_percentage': actual_similarity_percentage,
            'recovery_category_by_percent': recovery_category_by_percent,
            'recovery_category_by_stage': recovery_category_by_stage
        }
        return jsonify({'evaluation_results': evaluation_results}), 200
    except Exception as e:
        # Generic error handling
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
