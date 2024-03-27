import torch
from torch import nn
from torch.utils.data import DataLoader, random_split, TensorDataset
from model import MultimodalNetwork
from dataset import MultimodalDataset
import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate
import torch
from dataset import PreprocessTransform
from tqdm import tqdm
import os
import warnings
warnings.simplefilter('ignore')
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns  # For a nicer confusion matrix visualization

scaler = GradScaler()

# Checking for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = None

# class NormalizeTransform:
#     def __call__(self, img):
#         return (img - img.mean()) / img.std()


def pad_sequence(sequences):
    # Find the maximum size in the sequences
    max_size = max([s.size() for s in sequences])
    # Pad the sequences to the max size
    padded_sequences = [F.pad(s, (0, max_size[2] - s.size(2), 0, max_size[1] - s.size(1), 0, max_size[0] - s.size(0))) for s in sequences]
    return torch.stack(padded_sequences)

def my_collate_fn(batch):
    batch = {k: [d[k] for d in batch] for k in batch[0]}
    for k in batch:
        if k == 'image_data':  # Apply padding only to image data
            batch[k] = pad_sequence(batch[k])
        else:  # Use the default collate function for other data types
            batch[k] = default_collate(batch[k])
    return batch

def train_test_split(dataset, train_ratio=0.7):
    train_size = int(len(dataset) * train_ratio)
    test_size = len(dataset) - train_size
    return random_split(dataset, [train_size, test_size])

# Fetching the dataset
dataset = MultimodalDataset(csv_file='/home/arkaprabha/Documents/Alzheimer_Disease_Detection/Data/v1_dataset/ADNIMERGE_18Sep2023_final2.csv',
                            img_folder='/home/arkaprabha/Documents/Alzheimer_Disease_Detection/Data/images_Adni_final_v2',
                            genetic_folder_path = "/home/arkaprabha/Documents/Alzheimer_Disease_Detection/Data/v1_dataset/ADNI_Genetic_Merge",
                            transform=PreprocessTransform((64, 64, 64)))

# Splitting the dataset
train_dataset, test_dataset = train_test_split(dataset, train_ratio=0.7)
print(len(train_dataset))
# Creating DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=my_collate_fn, pin_memory=True, num_workers = 16)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=my_collate_fn, pin_memory=True, num_workers = 16)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultimodalNetwork(tabular_data_size=65, n_classes=3).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Creating synthetic dataset
batch_size = 32

results_dir = 'training_results'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Training loop
epochs = 200
# print(len(train_loader))
for epoch in range(epochs):
    loss_list = []
    correct_predictions = 0
    total_predictions = 0
    accuracy = 0
    avg_loss = 0
    # Training phase
    model.train()
    train_loss, train_correct, train_total = 0, 0, 0
    epoch_train_logits, epoch_train_targets = [], []
    # Initialize lists for accumulating labels and predictions across the epoch
    epoch_train_labels, epoch_train_predictions = [], []
    epoch_test_labels, epoch_test_predictions = [], []
    # Inside the training loop
    loop = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', leave=True)
    for batch in loop:
        tabular_data, image_data, genetic_data, labels = batch['tabular_data'].to(device), batch['image_data'].to(device), batch['genetic_data'].to(device), batch['label'].to(device)
        
        # Forward pass
        # print(tabular_data.shape, genetic_data.shape, image_data.shape)
        outputs = model(tabular_data, genetic_data, image_data)
        loss = criterion(outputs, torch.max(labels, 1)[1])

        epoch_train_logits.append(outputs.cpu().detach())
        epoch_train_targets.append(labels.cpu().detach())
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * tabular_data.size(0)
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == torch.max(labels, 1)[1]).sum().item()
        acc = 100 * train_correct / train_total
        train_precision = precision_score(torch.max(labels, 1)[1].cpu(), predicted.cpu(), average='macro')
        train_recall = recall_score(torch.max(labels, 1)[1].cpu(), predicted.cpu(), average='macro')
        train_f1 = f1_score(torch.max(labels, 1)[1].cpu(), predicted.cpu(), average='macro')
        loop.set_postfix(loss=loss.item(), accuracy=f'{acc:.2f}%')

        # Collect labels and predictions for confusion matrix
        epoch_train_labels.append(torch.max(labels, 1)[1].cpu())
        epoch_train_predictions.append(predicted.cpu())

    epoch_train_logits = torch.cat(epoch_train_logits)
    epoch_train_targets = torch.cat(epoch_train_targets)
    torch.save(epoch_train_logits, os.path.join(results_dir, f'train_logits_epoch_{epoch+1}.pt'))
    torch.save(epoch_train_targets, os.path.join(results_dir, f'train_targets_epoch_{epoch+1}.pt'))

    train_loss_avg = train_loss /train_total
    # After the training phase, concatenate lists of labels and predictions
    epoch_train_labels = torch.cat(epoch_train_labels)
    epoch_train_predictions = torch.cat(epoch_train_predictions)
    train_acc = 100* train_correct / train_total
    #  Evaluation phase
    model.eval()
    test_loss, test_correct, test_total = 0, 0, 0
    epoch_test_logits, epoch_test_targets = [], []
    # Inside the testing loop
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f' Testing ... Epoch {epoch+1}/{epochs}', leave=True):
            tabular_data, image_data, genetic_data, labels = batch['tabular_data'].to(device), batch['image_data'].to(device), batch['genetic_data'].to(device), batch['label'].to(device)
            
            # Forward pass
            outputs = model(tabular_data, genetic_data, image_data)
            loss = criterion(outputs, torch.max(labels, 1)[1])

            # Collecting logits and targets for analysis
            epoch_test_logits.append(outputs.cpu().detach())
            epoch_test_targets.append(labels.cpu().detach())
            
            test_loss += loss.item() * tabular_data.size(0)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == torch.max(labels, 1)[1]).sum().item()

            # Collect labels and predictions for confusion matrix
            epoch_test_labels.append(torch.max(labels, 1)[1].cpu())
            epoch_test_predictions.append(predicted.cpu())

    # At the end of each epoch, concatenate and save logits and targets for testing
    epoch_test_logits = torch.cat(epoch_test_logits)
    epoch_test_targets = torch.cat(epoch_test_targets)
    torch.save(epoch_test_logits, os.path.join(results_dir, f'test_logits_epoch_{epoch+1}.pt'))
    torch.save(epoch_test_targets, os.path.join(results_dir, f'test_targets_epoch_{epoch+1}.pt'))

    test_loss_avg = test_loss / test_total
    # After the testing phase, concatenate lists of labels and predictions
    epoch_test_labels = torch.cat(epoch_test_labels)
    epoch_test_predictions = torch.cat(epoch_test_predictions)
    test_acc = 100 * test_correct / test_total
    if (epoch) % 10 == 0:
        # Compute confusion matrices
        train_confusion_matrix = confusion_matrix(epoch_train_labels, epoch_train_predictions)
        test_confusion_matrix = confusion_matrix(epoch_test_labels, epoch_test_predictions)
        # Plot training confusion matrix
        plt.figure(figsize=(10, 7))
        sns.heatmap(train_confusion_matrix, annot=True, fmt="d", cmap="Blues")
        plt.title(f'Training Confusion Matrix - Epoch {epoch+1}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.savefig(f'train_confusion_matrix_epoch_{epoch+1}.png')
        plt.close()
        
        # Plot testing confusion matrix
        plt.figure(figsize=(10, 7))
        sns.heatmap(test_confusion_matrix, annot=True, fmt="d", cmap="Blues")
        plt.title(f'Testing Confusion Matrix - Epoch {epoch+1}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.savefig(f'test_confusion_matrix_epoch_{epoch+1}.png')
        plt.close()
    
    # Calculate metrics after every epoch
    train_precision = precision_score(epoch_train_labels, epoch_train_predictions, average='macro')
    train_recall = recall_score(epoch_train_labels, epoch_train_predictions, average='macro')
    train_f1 = f1_score(epoch_train_labels, epoch_train_predictions, average='macro')
    test_precision = precision_score(epoch_test_labels, epoch_test_predictions, average='macro')
    test_recall = recall_score(epoch_test_labels, epoch_test_predictions, average='macro')
    test_f1 = f1_score(epoch_test_labels, epoch_test_predictions, average='macro')
    
    
    print(f'Epoch {epoch+1}/{epochs}: '
      f'Train Loss: {train_loss_avg:.4f}, '
      f'Train Acc: {train_acc:.2f}%, '
      f'Train Precision: {train_precision:.4f}, '
      f'Train Recall: {train_recall:.4f}, '
      f'Train F1: {train_f1:.4f}, '
      f'Test Loss: {test_loss_avg:.4f}, '
      f'Test Acc: {test_acc:.2f}%, '
      f'Test Precision: {test_precision:.4f}, '
      f'Test Recall: {test_recall:.4f}, '
      f'Test F1: {test_f1:.4f}')

    # Writing to the file
    with open("statistics_v6.txt", "a") as file:
        file.write(f'Epoch {epoch+1}/{epochs}: '
                f'Train Loss: {train_loss_avg:.4f}, '
                f'Train Acc: {train_acc:.2f}%, '
                f'Train Precision: {train_precision:.4f}, '
                f'Train Recall: {train_recall:.4f}, '
                f'Train F1: {train_f1:.4f}, '
                f'Test Loss: {test_loss_avg:.4f}, '
                f'Test Acc: {test_acc:.2f}%, '
                f'Test Precision: {test_precision:.4f}, '
                f'Test Recall: {test_recall:.4f}, '
                f'Test F1: {test_f1:.4f}\n')

    # if (epoch + 1) % 20 == 0:
    checkpoint_filename = f'modelv6_checkpoint.pth'
    checkpoint_path = os.path.join(checkpoint_filename)
    torch.save(model.state_dict(), checkpoint_path)
    print(f"=====>Saved checkpoint: {checkpoint_path}")
    
# Testing the model with a single batch
# with torch.no_grad():
#     tabular_data, image_data, labels = next(iter(dataloader))
#     outputs = model(tabular_data, image_data)
    # Calculate the accuracy

