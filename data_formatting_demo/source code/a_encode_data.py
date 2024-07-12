import json
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

file_name = "encoded_raw_data.csv"

def read_config(config_file):
    with open(config_file, 'r') as file:
        config = json.load(file)
    return config

def load_and_encode_data(raw_data_csv_file):
    # Load data
    data = pd.read_csv(raw_data_csv_file)

    # Label encoding
    le_race = LabelEncoder()
    data['race'] = le_race.fit_transform(data['race'])
    print("Race Mapping:", dict(zip(le_race.classes_, range(len(le_race.classes_)))))

    le_gender = LabelEncoder()
    data['gender'] = le_gender.fit_transform(data['gender'])
    print("Gender Mapping:", dict(zip(le_gender.classes_, range(len(le_gender.classes_)))))


    le_diabetesMed = LabelEncoder()
    data['diabetesMed'] = le_diabetesMed.fit_transform(data['diabetesMed'])
    print("DiabetesMed Mapping:", dict(zip(le_diabetesMed.classes_, range(len(le_diabetesMed.classes_)))))


    le_admission_source = LabelEncoder()
    data['admission_source_id'] = le_admission_source.fit_transform(data['admission_source_id'])
    print("Admission Source ID Mapping:", dict(zip(le_admission_source.classes_, range(len(le_admission_source.classes_)))))


    # Custom mappings
    age_mapping = {k: i for i, k in enumerate(['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)', '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)'])}
    data['age'] = data['age'].map(age_mapping)
    print("Age Mapping:", age_mapping)

    hba1c_mapping = {'Norm': 0, '>7': 1, '>8': 2, '>9': 3}
    data['HbA1c'] = data['HbA1c'].map(hba1c_mapping)
    print("HbA1c Mapping:", hba1c_mapping)

    # Encode 'readmitted'
    data['readmitted'] = data['readmitted'].apply(lambda x: 0 if x == 'NO' else 1)

    return data


def generate_csv(data, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    encoded_file = os.path.join(file_path, file_name)
    data.to_csv(encoded_file, index=False)
    print(encoded_file, "is generated, ", "please check and verify")

def main():
    config = read_config('data_prepare_config.json')
    raw_data_csv_file = config['raw_data_csv_file']
    input_data_path = config['input_data_path']
    # Load and encode data
    data = load_and_encode_data(raw_data_csv_file)
    generate_csv(data, input_data_path)
    
if __name__ == "__main__":
    main()




