import json
import pandas as pd
import os
from a_encode_data import file_name




def read_config(config_file):
    with open(config_file, 'r') as file:
        config = json.load(file)
    return config


def generate_header_and_encoded_csv(input_data_path, header_file, encoded_data_file, label_name):

    # Read the cleaned data from the input_data_path
    data = pd.read_csv(os.path.join(input_data_path, file_name))

    # Move 'readmitted' column to the first position
    data = data[[label_name] + [col for col in data.columns if col != label_name]]
    
    # Create the directory if it doesn't exist
    if not os.path.exists(input_data_path):
        os.makedirs(input_data_path)
        print("Directory created successfully!")
        
    # Get the column names from the data
    column_names = data.columns.tolist()
        
    # Write the column names to the header file
    with open(header_file, 'w') as file:
        file.write(','.join(column_names))
    print(header_file, "is generated successfully!")

    # Save the modified data to the same file
    data.to_csv(encoded_data_file, index=False, header=False)
    print(encoded_data_file, "is generated successfully!")
        
    

def main():
    config = read_config('data_prepare_config.json')
    label_name = config['label_name']
    input_data_path = config['input_data_path']
    header_file = os.path.join(config['input_data_path'], config['headers_datasheet_name'])
    encoded_data_file = os.path.join(config['input_data_path'], config['encoded_datasheet_name'])

    # Generate directory, header and encoded data 
    generate_header_and_encoded_csv(input_data_path, header_file, encoded_data_file, label_name)
    
if __name__ == "__main__":
    main()




