#!/usr/bin/env python3
"""
Pneumonia Classification CNN Model
==================================

A convolutional neural network for classifying chest X-rays as NORMAL or PNEUMONIA.
This script includes data preprocessing, model training, and evaluation with proper
data leakage prevention for patient-based splits.
"""
#Lucian Irsigler 2621933, Prashan Rajaratnam 2436566, Banzile Nhlebela 2571291, Pramit Kanji 2551233
import os
import re
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import (
    Dataset,
    DataLoader,
    random_split,
    Subset,
    WeightedRandomSampler
)
from torch.nn.utils import clip_grad_norm_

import torchvision.transforms as transforms
from torchvision import datasets


# ============================================================================
# HYPERPARAMETERS AND CONFIGURATION
# ============================================================================

HYPERPARAMETERS = {
    "num_epochs": 10,
    "batch_size": 64,
    "image_size": 56,
    "lr": 0.001,
    "size": (0.6, 0.2, 0.2),  # train-val-test split
    "balanced": True
}

CLASSES = ("NORMAL", "PNEUMONIA")


# ============================================================================
# DATA LEAKAGE PREVENTION UTILITIES
# ============================================================================

def extract_patient_ids(filename):
    """Extract patient ID from filename to prevent data leakage."""
    patient_id = filename.split('_')[0].replace("person", "")
    return patient_id


def split_file_names(hyperparams, folder="chest", seed=42):
    """
    Split PNEUMONIA files by patient ID to prevent data leakage.
    Ensures that all images from the same patient are in the same dataset split.
    """
    random.seed(seed)
    location = os.path.join(folder, "PNEUMONIA")

    # Get unique patient IDs
    pneumonia_patient_ids = list(set([extract_patient_ids(fn) for fn in os.listdir(location)]))
    total_patients = len(pneumonia_patient_ids)

    # Shuffle and split
    random.shuffle(pneumonia_patient_ids)
    train_cutoff = int(hyperparams["size"][0] * total_patients)
    val_cutoff = int((1 - hyperparams["size"][1]) * total_patients)

    train_ids = set(pneumonia_patient_ids[:train_cutoff])
    val_ids = set(pneumonia_patient_ids[train_cutoff:val_cutoff])
    test_ids = set(pneumonia_patient_ids[val_cutoff:])

    # Initialize file path lists
    train_files, val_files, test_files = [], [], []

    for fn in os.listdir(location):
        patient_id = extract_patient_ids(fn)
        full_path = os.path.join(location, fn)

        if patient_id in train_ids:
            train_files.append(full_path)
        elif patient_id in val_ids:
            val_files.append(full_path)
        elif patient_id in test_ids:
            test_files.append(full_path)

    return train_files, val_files, test_files


def check_data_leakage(train_loader, val_loader, test_loader):
    """Check for data leakage between train/val/test sets for PNEUMONIA patients."""
    location = os.path.join("chest", "PNEUMONIA")
    pneumonia_files = os.listdir(location)
    pneumonia_patient_ids = list(set([extract_patient_ids(fn) for fn in pneumonia_files]))
    total_patients = len(pneumonia_patient_ids)
    print(f"Total unique PNEUMONIA patients: {total_patients}")

    leaked = False

    for patient_id in pneumonia_patient_ids:
        pattern = re.compile(rf'person{patient_id}(?!\d)')

        train_files = [path for path in train_loader.dataset.image_paths if pattern.search(path)]
        val_files = [path for path in val_loader.dataset.image_paths if pattern.search(path)]
        test_files = [path for path in test_loader.dataset.image_paths if pattern.search(path)]

        if sum(map(bool, [train_files, val_files, test_files])) > 1:
            print(f"\nData leakage detected for patient ID: {patient_id}")
            print(f"  → In train: {len(train_files)} file(s)")
            for f in train_files:
                print(f"    - {f}")
            print(f"  → In val: {len(val_files)} file(s)")
            for f in val_files:
                print(f"    - {f}")
            print(f"  → In test: {len(test_files)} file(s)")
            for f in test_files:
                print(f"    - {f}")
            leaked = True

    if not leaked:
        print("No data leak present")
    return leaked


# ============================================================================
# CUSTOM DATASET CLASS
# ============================================================================

class CustomImageDataset(Dataset):
    """Custom dataset class for handling image paths with automatic label inference."""
    
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        self.label_map = {"NORMAL": 0, "PNEUMONIA": 1}
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        path = self.image_paths[idx]
        
        # Infer label from file path
        if "NORMAL" in path:
            label = self.label_map["NORMAL"]
        elif "PNEUMONIA" in path:
            label = self.label_map["PNEUMONIA"]
        else:
            raise ValueError(f"Unknown class in file path: {path}")
        
        # Load image
        try:
            image = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            # Return a blank image as fallback
            image = Image.new('RGB', (224, 224), color='black')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        
        return image, label
    
    def get_indices(self):
        """Return all valid indices for this dataset"""
        return list(range(len(self.image_paths)))
    
    def get_image_label_pairs(self):
        """Return list of tuples (img_path, class_label) for all images"""
        pairs = []
        for path in self.image_paths:
            if "NORMAL" in path:
                label = self.label_map["NORMAL"]
            elif "PNEUMONIA" in path:
                label = self.label_map["PNEUMONIA"]
            else:
                raise ValueError(f"Unknown class in file path: {path}")
            pairs.append((path, label))
        return pairs


# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def create_weighted_sampler(dataset):
    """Create a weighted sampler to handle class imbalance."""
    targets = [label for _, label in dataset]
    class_counts = np.bincount(targets)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum()
    weights = [class_weights[label] for label in targets]
    sampler = WeightedRandomSampler(weights, len(weights))
    return sampler


def load_data(folder="chest"):
    """Load data using PyTorch's ImageFolder."""
    transform = transforms.Compose([
        transforms.Resize((HYPERPARAMETERS["image_size"], HYPERPARAMETERS["image_size"])),
        transforms.ToTensor()
    ])
    full_dataset = datasets.ImageFolder(folder, transform=transform)
    return full_dataset


def split_data(dataset, hyperparams):
    """Split dataset into train/val/test sets."""
    if sum(hyperparams["size"]) != 1.0:
        raise ValueError("Size proportions must sum to 1.0")
    
    train_size = int(hyperparams["size"][0] * len(dataset))
    val_size = int(hyperparams["size"][1] * len(dataset))
    test_size = len(dataset) - val_size - train_size

    train_data, val_data, test_data = random_split(dataset, [train_size, val_size, test_size])
    return train_data, val_data, test_data


def get_normal_indices(dataset):
    """Get indices for NORMAL class images."""
    normal_indices = [i for i, (path, label) in enumerate(dataset.imgs) if "NORMAL" in path]
    return normal_indices


def split_normal_data(dataset, indices):
    """Split NORMAL data using standard random split."""
    normal_dataset = Subset(dataset, indices)
    train_data, val_data, test_data = split_data(normal_dataset, HYPERPARAMETERS)
    return train_data, val_data, test_data


def get_normal_files(dataset, train_data, val_data, test_data):
    """Extract file paths from split NORMAL data."""
    train_files = [dataset.imgs[idx][0] for idx in train_data.indices]
    val_files = [dataset.imgs[idx][0] for idx in val_data.indices]
    test_files = [dataset.imgs[idx][0] for idx in test_data.indices]
    return train_files, val_files, test_files


def combine_files(normal_files, pneumonia_files):
    """Combine NORMAL and PNEUMONIA file lists."""
    if len(normal_files) != len(pneumonia_files):
        raise ValueError("normal_files and pneumonia_files must match in size")

    final_test_files = normal_files[0] + pneumonia_files[0]
    final_val_files = normal_files[1] + pneumonia_files[1]
    final_train_files = normal_files[2] + pneumonia_files[2]

    return final_test_files, final_val_files, final_train_files


def create_data_loaders(train_data, val_data, test_data, batch_size, balanced=False):
    """Create DataLoaders for train/val/test sets."""
    if balanced:
        sampler = create_weighted_sampler(train_data)
        train_loader = DataLoader(train_data, 
                    batch_size=batch_size, 
                    sampler=sampler, 
                    num_workers=4)
    else: 
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, test_loader


def load_pipeline(hyperparams):
    """Complete data loading pipeline with leakage prevention."""
    dataset = load_data()

    # NORMAL data processing
    normal_indices = get_normal_indices(dataset)
    train_data, val_data, test_data = split_normal_data(dataset, normal_indices)
    n_train_files, n_val_files, n_test_files = get_normal_files(dataset, train_data, val_data, test_data)

    # PNEUMONIA data processing (with patient-based splitting)
    p_train_files, p_val_files, p_test_files = split_file_names(hyperparams)

    train_files, val_files, test_files = combine_files(
        [n_train_files, n_val_files, n_test_files],
        [p_train_files, p_val_files, p_test_files]
    )
    
    # Create custom datasets
    transform = transforms.Compose([
        transforms.Resize((hyperparams["image_size"], hyperparams["image_size"])),
        transforms.ToTensor()
    ])

    train_dataset = CustomImageDataset(train_files, transform=transform)
    val_dataset = CustomImageDataset(val_files, transform=transform)
    test_dataset = CustomImageDataset(test_files, transform=transform)

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, val_dataset, test_dataset, 
        hyperparams["batch_size"], hyperparams["balanced"]
    )
    
    return train_loader, val_loader, test_loader


# ============================================================================
# DATA ANALYSIS AND VISUALIZATION
# ============================================================================

def visualize_class_distribution(loader, dataset_name):
    """Visualize class distribution in a dataset."""
    class_counts = np.zeros(2)
    for _, y in loader.dataset.get_image_label_pairs():
        class_counts[y] += 1

    print([(CLASSES[i], class_counts[i]) for i in range(len(CLASSES))])
    plt.figure(figsize=(8, 6))
    plt.bar(["Normal", "Pneumonia"], class_counts)
    plt.xlabel("Classes")
    plt.ylabel("Count")
    plt.title(f"Class distribution in {dataset_name} data")
    plt.show()


def calculate_stats(loader):
    """Calculate mean and std statistics for a dataset."""
    mean = 0.
    std = 0.
    total_images = 0

    for images, _ in tqdm(loader, desc="Calculating stats"):
        batch_size = images.size(0)
        images = images.view(batch_size, images.size(1), -1)  # (B, C, H*W)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images += batch_size

    mean /= total_images
    std /= total_images

    return mean.numpy(), std.numpy()


def analyze_datasets(train_loader, val_loader, test_loader):
    """Analyze and visualize all datasets."""
    print("=== DATASET ANALYSIS ===")
    
    visualize_class_distribution(train_loader, "Train")
    mean, std = calculate_stats(train_loader)
    print(f"Train dataset mean and std: {mean}, {std}")

    visualize_class_distribution(val_loader, "Validation")
    mean1, std1 = calculate_stats(val_loader)
    print(f"Validation dataset mean and std: {mean1}, {std1}")

    visualize_class_distribution(test_loader, "Test")
    mean2, std2 = calculate_stats(test_loader)
    print(f"Test dataset mean and std: {mean2}, {std2}")

    return [mean, mean1, mean2], [std, std1, std2]


# ============================================================================
# CNN MODEL ARCHITECTURE
# ============================================================================

class CNN(nn.Module):
    """
    Convolutional Neural Network for pneumonia classification.
    
    Architecture:
    - 3 Convolutional layers with BatchNorm, ReLU, MaxPool, Dropout2D
    - AdaptiveAvgPool2d for fixed-size output
    - 2 Fully connected layers with Dropout
    - Output: 2 classes (NORMAL, PNEUMONIA)
    """
    
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.dropout2d = nn.Dropout2d(0.3)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))

        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout2d(x)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout2d(x)
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout2d(x)

        x = self.adaptive_pool(x)

        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        x = self.fc2(x)
        return x


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def validation_loss(model, val_loader):
    """Calculate validation loss."""
    loss_function = torch.nn.functional.cross_entropy
    total_loss = 0.0
    model.eval()

    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            total_loss += loss
    
    return total_loss / len(val_loader)


def plot_training_history(train_loss_history, val_loss_history):
    """Plot training and validation loss history."""
    temp_train = [i.item() if torch.is_tensor(i) else i for i in train_loss_history]
    temp_val = [i.item() if torch.is_tensor(i) else i for i in val_loss_history]
    
    plt.figure(figsize=(10, 6))
    plt.plot(temp_train, label="Train loss")
    plt.plot(temp_val, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.show()


def train_model(model, train_loader, val_loader, criterion, optimizer, hyperparams):
    """Train the CNN model."""
    n_total_steps = len(train_loader)
    train_loss_history = []
    val_loss_history = []

    print("=== STARTING TRAINING ===")
    
    for epoch in range(hyperparams["num_epochs"]):
        model.train()
        running_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{hyperparams['num_epochs']}", 
                            leave=False, unit="batch")

        for i, (images, labels) in enumerate(progress_bar):
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping to avoid exploding gradients
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()

            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{hyperparams['num_epochs']}], "
                      f"Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}")

        avg_train_loss = running_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)

        val_loss = validation_loss(model, val_loader)
        val_loss_history.append(val_loss)

        print(f"Epoch [{epoch+1}/{hyperparams['num_epochs']}], "
              f"Train Loss: {avg_train_loss:.4f}, Validation Loss: {val_loss:.4f}")

    print("Finished Training")
    return train_loss_history, val_loss_history


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def evaluate_model(model, test_loader, classes):
    """Evaluate the trained model on test data."""
    model.eval()
    true_labels = []
    predicted_labels = []
    correct = 0
    total = 0

    print("=== EVALUATING MODEL ===")
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(test_loader, desc="Evaluating")):
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)
    class_report = classification_report(true_labels, predicted_labels, target_names=classes)

    print(f"\n=== EVALUATION RESULTS ===")
    print(f"Accuracy on the test set: {accuracy:.2%}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("\nClassification Report:")
    print(class_report)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "report": class_report,
        "true_labels": true_labels,
        "predicted_labels": predicted_labels
    }


def plot_confusion_matrix(metrics):
    """Plot confusion matrix."""
    cm = confusion_matrix(metrics["true_labels"], metrics["predicted_labels"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["NORMAL", "PNEUMONIA"])
    plt.figure(figsize=(8, 6))
    disp.plot()
    plt.title("Confusion Matrix")
    plt.show()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("=== PNEUMONIA CLASSIFICATION CNN ===")
    print(f"Hyperparameters: {HYPERPARAMETERS}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Load and prepare data
    print("\n=== LOADING DATA ===")
    train_loader, val_loader, test_loader = load_pipeline(HYPERPARAMETERS)
    
    # Check for data leakage
    print("\n=== CHECKING DATA LEAKAGE ===")
    check_data_leakage(train_loader, val_loader, test_loader)
    
    # Analyze datasets
    print("\n=== ANALYZING DATASETS ===")
    means, stds = analyze_datasets(train_loader, val_loader, test_loader)
    
    # Initialize model, loss, and optimizer
    print("\n=== INITIALIZING MODEL ===")
    model = CNN()
    
    # Calculate class weights for balanced loss
    targets = [label for _, label in train_loader.dataset.get_image_label_pairs()]
    class_counts = np.bincount(targets)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum()
    class_weights_tensor = torch.FloatTensor(class_weights)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=HYPERPARAMETERS["lr"], weight_decay=1e-4)
    
    print(f"Model architecture:\n{model}")
    print(f"Class weights: {class_weights}")
    
    # Train the model
    train_loss_history, val_loss_history = train_model(
        model, train_loader, val_loader, criterion, optimizer, HYPERPARAMETERS
    )
    
    # Plot training history
    plot_training_history(train_loss_history, val_loss_history)
    
    # Evaluate the model
    metrics = evaluate_model(model, test_loader, CLASSES)
    
    # Plot confusion matrix
    plot_confusion_matrix(metrics)
    
    # Save the model
    model_path = "pneumonia_cnn_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'hyperparameters': HYPERPARAMETERS,
        'metrics': metrics,
        'class_weights': class_weights
    }, model_path)
    print(f"\nModel saved to {model_path}")
    
    return model, metrics


if __name__ == "__main__":
    model, metrics = main()