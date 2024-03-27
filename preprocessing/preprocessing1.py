import os
import shutil

# Define the master folder and the destination folder
master_folder = 'ADNI'  # Replace with the path to your master folder
destination_folder = 'images_Adni_final_v2'  # Replace with the path to your destination folder

# Create the destination folder if it does not exist
os.makedirs(destination_folder, exist_ok=True)

# Walk through the master folder
for root, dirs, files in os.walk(master_folder):
    for file in files:
        # Check for .nii files
        if file.endswith('.nii'):
            # Construct the full file path
            file_path = os.path.join(root, file)
            # Construct the destination file path
            destination_file_path = os.path.join(destination_folder, file)
            
            # Move the .nii file to the destination folder
            shutil.move(file_path, destination_file_path)
            print(f'Moved file {file_path} to {destination_file_path}')

print("Finished moving .nii files.")
