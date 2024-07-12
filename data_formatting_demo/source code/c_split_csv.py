import json
import os
import shutil
import pandas as pd


def read_config(config_file):
    with open(config_file, 'r') as file:
        config = json.load(file)
    return config

def load_data(encoded_data_file) -> pd.DataFrame:
    # Read the CSV file into a pandas DataFrame
    return pd.read_csv(encoded_data_file, header=None)


def split_csv(encoded_data_file, output_dir, num_parts, part_name, sample_rate):
    df = load_data(encoded_data_file)

    # Calculate the number of rows per part
    total_size = int(len(df) * sample_rate)
    rows_per_part = total_size // num_parts

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Split the DataFrame into N parts
    for i in range(num_parts):
        start_index = i * rows_per_part
        end_index = (i + 1) * rows_per_part if i < num_parts - 1 else total_size
        print(f"{part_name}{i + 1}=", f"{start_index=}", f"{end_index=}")
        part_df = df.iloc[start_index:end_index]

        # Save each part to a separate CSV file
        output_file = os.path.join(output_dir, f"{part_name}{i + 1}.csv")
        part_df.to_csv(output_file, header=False, index=False)
        print(f"File copied to {output_file}")


def distribute_header_file(header_file: str, output_dir: str, num_parts: int, part_name: str):

    # Split the DataFrame into N parts
    for i in range(num_parts):
        output_file = os.path.join(output_dir, f"{part_name}{i + 1}_header.csv")
        shutil.copy(header_file, output_file)
        print(f"File copied to {output_file}")


def main():
    config = read_config('data_prepare_config.json')
    header_file = os.path.join(config['input_data_path'], config['headers_datasheet_name'])
    encoded_data_file = os.path.join(config['input_data_path'], config['encoded_datasheet_name'])
    output_directory = config['output_dir']
    num_parts = int(config['site_num'])
    site_name_prefix = config['site_name_prefix']
    sample_rate = float(config['sample_rate'])
    split_csv(encoded_data_file, output_directory, num_parts, site_name_prefix, sample_rate)
    distribute_header_file(header_file, output_directory, num_parts, site_name_prefix)


if __name__ == "__main__":
    main()
