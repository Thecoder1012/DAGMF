import multiprocessing as mp
import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import numpy as np
import warnings 
warnings.filterwarnings('ignore')

# Define the function that will be run by each worker
def process_ptid(ptid, main_df, genetic_folder_path, new_genetic_folder_path):
    for subdir, _, files in os.walk(genetic_folder_path):
        for filename in files:
            if ptid in filename:
                full_path = os.path.join(subdir, filename)
                genetic_df = read_csv_exclude_negative(full_path)
                sampled_df = genetic_df.sample(n=min(500, len(genetic_df)), random_state=42)
                allele_columns = [col for col in sampled_df.columns if col.startswith('Allele')]
                sampled_df = sampled_df[allele_columns]
                new_path = os.path.join(new_genetic_folder_path, Path(subdir).relative_to(genetic_folder_path), filename)
                Path(new_path).parent.mkdir(parents=True, exist_ok=True)
                sampled_df.to_csv(new_path, index=False)
                break
    cols_to_drop = ['RID', 'COLPROT', 'ORIGPROT', 'SITE', 'VISCODE', 'EXAMDATE', 'update_stamp']
    main_df = main_df.drop(columns=cols_to_drop, errors='ignore')
    main_df = main_df.drop(columns=main_df.filter(regex='^(MOCA|Ecog)').columns)
    # Drop columns with more than 70% missing values
    thresh = len(main_df) * 0.30
    main_df = main_df.dropna(thresh=thresh, axis=1)
    columns_with_nan = main_df.columns[main_df.isna().any()].tolist()
    
    #fill empty values
    for col in main_df.select_dtypes(include=['number']).columns:
        main_df[col].fillna(main_df[col].mean(), inplace=True)

    # Save the preprocessed main CSV file
    main_df.to_csv(main_csv_path.replace('.csv', '_preprocessed1.csv'), index=False)
    # Load the CSV file
    # Fill missing values for numerical columns with their mean
    numerical_cols = main_df.select_dtypes(include=['number']).columns
    main_df[numerical_cols] = main_df[numerical_cols].fillna(main_df[numerical_cols].mean())

    # Fill missing values for categorical columns with their mode
    # Here, we also make sure to exclude any string values that start with '>'
    categorical_cols = main_df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        # Exclude any entries that start with '>'
        mode_value = main_df[col][~main_df[col].str.startswith('>', na=False)].mode()[0]
        main_df[col] = main_df[col].fillna(mode_value)

    # Save the filled DataFrame back to a CSV file
    main_df.to_csv('Stage2/ADNIMERGE_26Sep2023_final1.csv', index=False)
    columns_with_nan = main_df.columns[main_df.isna().any()].tolist()

def Preprocess_Genetic(main_csv_path, genetic_folder_path, new_genetic_folder_path):
    main_df = pd.read_csv(main_csv_path)
    Path(new_genetic_folder_path).mkdir(parents=True, exist_ok=True)
    
    ptids = main_df['PTID'].unique()
    
    # Set up a pool of workers
    pool = mp.Pool(mp.cpu_count())
    
    # Create a list of tasks for the pool
    tasks = [(ptid, main_df, genetic_folder_path, new_genetic_folder_path) for ptid in ptids]
    
    # Map the function onto the tasks
    pool.starmap(process_ptid, tasks)
    
    # Close the pool and wait for all tasks to complete
    pool.close()
    pool.join()

def read_csv_exclude_negative(full_path):
    # Temporary list to store rows that meet the criteria
    valid_rows = []
    with open(full_path, 'r') as file:
        headers = file.readline().strip()  # Read the first line as headers
        for line in file:
            if not any(value.strip().startswith('-') for value in line.split(',')):
                valid_rows.append(line.strip())

    # Convert the list of valid rows into a DataFrame
    df = pd.DataFrame([row.split(',') for row in valid_rows], columns=headers.split(','))
    return df

def check_images_exist(genetic_data_path, images_folder_path, main_csv_path):
    # Load the CSV
    ptids = []
    df = pd.read_csv(main_csv_path)
    for root, dirs, files in os.walk(genetic_data_path):
        for file in files:
            # Check if the file is a CSV
            if file.endswith('.csv'):
                # Construct the full path and add it to the list
                full_path = os.path.join(root, file)
                ptids.append(file.split('.')[0])
    
    # Join the list elements separated by a comma
    imageuid_str = ','.join(map(str, ptids))

    # Define the output file path
    output_txt_path = 'ptids.txt'  # Replace with your desired output file path

    # Write the string to a text file
    with open(output_txt_path, 'w') as file:
        file.write(imageuid_str)

    print(f"The list of IMAGEUIDs has been saved to {output_txt_path}")

    # Extract PTIDs from the CSV
    # ptids = df['PTID'].unique()
    print("ptid len: ", len(ptids))
    # Get a list of all image files in the images folder
    image_files = os.listdir(images_folder_path)
    print("image files len:", len(image_files))
    # Check if each PTID has a corresponding image file and collect the ones that don't
    missing_images = []
    for ptid in ptids:
        # Construct the expected start of the filename for the PTID
        expected_filename_start = f"ADNI_{ptid}"
        # Check if an       y image file starts with the expected filename start
        if not any(file.startswith(expected_filename_start) for file in image_files):
            missing_images.append(int(df.loc[df['PTID'] == ptid, 'IMAGEUID'].values[0]))
            # missing_images.append(ptid)
    
    # Join the list elements separated by a comma
    imageuid_str = ','.join(map(str, missing_images))

    # Define the output file path
    output_txt_path = 'missing_ptids.txt'  # Replace with your desired output file path

    # Write the string to a text file
    with open(output_txt_path, 'w') as file:
        file.write(imageuid_str)

    print(f"The list of IMAGEUIDs has been saved to {output_txt_path}")

    print("len of missing images: ",len(missing_images))
    return missing_images

# Call this from your main if running as a script
if __name__ == '__main__':
    main_csv_path = 'path to ADNIMERGE_18Sep2023.csv'
    genetic_folder_path = 'genetic folder path'
    new_genetic_folder_path = 'preprocessed genetic folder path'
    # images_folder_path = "images_Adni_final"
    images_folder_path = ".nii images path"
    # preprocessed_csv = "Stage2/ADNIMERGE_18Sep2023_final1.csv"
    Preprocess_Genetic(main_csv_path, genetic_folder_path, new_genetic_folder_path)
    check_images_exist_new(new_genetic_folder_path, images_folder_path, main_csv_path)
