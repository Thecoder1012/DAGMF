import torch
import pandas as pd
import numpy as np
import nibabel as nib
import os
import fnmatch
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from torch.utils.data import Dataset
import torchio as tio
from sklearn.impute import SimpleImputer
from torch.utils.data import TensorDataset
from sklearn.preprocessing import LabelEncoder
import warnings 
warnings.filterwarnings('ignore')

class PreprocessTransform:
    def __init__(self, new_shape):
        self.new_shape = new_shape  # New shape as a tuple (D, H, W)

    def __call__(self, img):
        # print("Hii:", img.shape)
        # Resize the image to new_shape
        resize_transform = tio.Resize(self.new_shape)
        img_resized = resize_transform(img)
        
        # Normalize the resized image
        img_normalized = (img_resized - img_resized.mean()) / img_resized.std()
        # print("hello:",img_normalized.shape)
        return img_normalized
        
class MultimodalDataset(Dataset):
    def __init__(self, csv_file, img_folder, genetic_folder_path, transform=None):
        self.tabular_frame = pd.read_csv(csv_file)
        # self.tabular_frame = self.tabular_frame.drop(columns=['IMAGEUID', 'IMAGEUID_bl'], errors='ignore')
        self.img_folder = img_folder
        self.transform = transform
        self.genetic_folder_path = genetic_folder_path
        # Preprocess the tabular data and get feature size
        # Filter for valid PTIDs
        self.valid_ptids = self.filter_valid_ptids()

        # Only keep entries from tabular_frame with valid PTIDs
        self.tabular_frame = self.tabular_frame[self.tabular_frame['PTID'].isin(self.valid_ptids)]

        self.features, self.target, self.feature_size = self.preprocess_tabular_data()
        # print(self.target)
        
        # Target variable
        # self.labels = torch.tensor(self.tabular_frame['DX'].values)

    def filter_valid_ptids(self):
        valid_ptids = []
        for _, row in self.tabular_frame.iterrows():
            ptid = row['PTID']
            genetic_path = os.path.join(self.genetic_folder_path, f"{ptid}.csv")
            image_path = self.find_image_file(ptid)  # Adjust find_image_file to return None instead of raising FileNotFoundError
            
            # img_path = self.find_image_file(self.tabular_frame.iloc[idx]['PTID'])
            if os.path.exists(genetic_path) and image_path:
                valid_ptids.append(ptid)
        return valid_ptids

    '''
    def preprocess_tabular_data(self):
        # Define columns to be scaled and encoded
        columns_to_scale = ['AGE', 'PTEDUCAT']
        columns_to_encode = ['PTGENDER', 'PTRACCAT', 'PTMARRY']

        # Preprocessing pipelines
        numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
        categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

        # Column transformer
        preprocessor = ColumnTransformer(transformers=[
            ('num', numeric_transformer, columns_to_scale),
            ('cat', categorical_transformer, columns_to_encode)])

        # Apply preprocessing
        features = preprocessor.fit_transform(self.tabular_frame)
        feature_size = features.shape[1]
        features = torch.tensor(features, dtype=torch.float32)

        return features, feature_size
    '''
    def preprocess_tabular_data(self):
        # First, we remove 'IMAGEUID' and 'IMAGEUID_bl' as they are not needed
        features = self.tabular_frame.drop(columns=['IMAGEUID', 'IMAGEUID_bl', 'EXAMDATE_bl'])
        # labels = self.tabular_frame['DX']

        # Define the target and feature columns
        target_column = 'DX'
        feature_columns = [col for col in features.columns if col != target_column]

        ptid = self.tabular_frame['PTID']
        # Define columns with discrete categorical data
        categorical_columns = ['PTID', 'DX_bl', 'PTGENDER', 'PTETHCAT', 'PTRACCAT', 'PTMARRY', 'APOE4', 'FSVERSION', 'FLDSTRENG', 'FSVERSION_bl', 'FLDSTRENG_bl']
        
        # Encode categorical columns
        for col in categorical_columns:
            if col in feature_columns:
                features[col] = LabelEncoder().fit_transform(features[col])
        
        # Scale numerical features
        numerical_columns = list(set(feature_columns) - set(categorical_columns))
        scaler = StandardScaler()
        features[numerical_columns] = scaler.fit_transform(features[numerical_columns])

        # One-hot encode the target
        target = pd.get_dummies(features[target_column], prefix='DX')
        
        # Separate features and target
        X = features.drop(columns=[target_column])
        y = target.values
        
        # Convert features and target to PyTorch tensors
        features_t = torch.tensor(X.values, dtype=torch.float32)
        labels = torch.tensor(y, dtype=torch.float32)  # Use torch.float32 if using BCEWithLogitsLoss
        
        feature_size = features_t.shape[1]
        
        return features_t, labels, feature_size

    def preprocess_genetic_data(self, df):
        # Replace nucleotides with numerical values
        nucleotide_to_number = {'A': 1, 'G': 2, 'C': 3, 'T': 4}
        for column in df.columns:
            df[column] = df[column].apply(lambda x: nucleotide_to_number[x] if x in nucleotide_to_number else np.random.randint(1, 5))

        # No need to one-hot encode, so we can directly standardize the data
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(df)
        scaled_df = pd.DataFrame(scaled_features, columns=df.columns)

        # scaled_df is now preprocessed and ready for use in an ANN
        return scaled_df

    def load_genetic_data(self, ptid):
        genetic_file_path = os.path.join(self.genetic_folder_path, f"{ptid}.csv")
        genetic_df = pd.read_csv(genetic_file_path)
        # print(genetic_df.head(1))
        scaled_gen_df = self.preprocess_genetic_data(genetic_df)
        genetic_data = torch.tensor(scaled_gen_df.values, dtype = torch.float32)
        
        return genetic_data

    def __len__(self):
        return len(self.tabular_frame)

    def __getitem__(self, idx):
        # Tabular data
        tabular_data = self.features[idx]

        # Image data loading with the corrected naming convention
        ptid = self.tabular_frame.iloc[idx]['PTID']
        # print(ptid)
        # img_path = self.find_image_file(ptid)
        img_path = self.find_image_file(self.tabular_frame.iloc[idx]['PTID'])
        # img_path = self.find_image_file(ptid)
        if img_path:
            img = tio.ScalarImage(img_path)
            # print(img.shape, "yeah!")
            if self.transform:
                img_data = self.transform(img.data)
                # print(img_data.shape,"hiirufhbkjcwr")
            else:
                img_data = img.data
            img_data = torch.tensor(img_data, dtype=torch.float32)
        else:
            # Define how you want to handle missing image data, e.g., zeros tensor
            # print("MiSSING DATA!!!!!!")
            img_data = torch.zeros(1, 64, 32, 32)  # Adjust the shape as necessary
        
        genetic_data = self.load_genetic_data(ptid)
        # print(tabular_data.shape, img_data.shape, genetic_data.shape, self.target[idx].shape)
        return {'tabular_data': tabular_data, 'image_data': img_data, 'genetic_data': genetic_data, 'label': self.target[idx]}
        
    def find_image_file(self, ptid):
        pattern = f'ADNI_{ptid}_*.nii'
        for file in os.listdir(self.img_folder):
            if fnmatch.fnmatch(file, pattern):
                return os.path.join(self.img_folder, file)
        return None  # Return None if no file matches

# Example of using the dataset
dataset = MultimodalDataset(csv_file='/home/arkaprabha/Documents/Alzheimer_Disease_Detection/Data/v1_dataset/ADNIMERGE_18Sep2023_final2.csv',
                            img_folder='/home/arkaprabha/Documents/Alzheimer_Disease_Detection/Data/images_Adni_final_v2',
                            genetic_folder_path = "/home/arkaprabha/Documents/Alzheimer_Disease_Detection/Data/v1_dataset/ADNI_Genetic_Merge",
                            transform=PreprocessTransform((128, 128, 128)))
# print(dataset[123]['tabular_data'].shape, dataset[123]['genetic_data'].shape, dataset[123]['image_data'].shape, dataset[123]['label'].shape)  # Example usage
