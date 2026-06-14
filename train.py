import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import matplotlib.pyplot as plt
import numpy as np

# Import our custom model
from model import get_model

# Configuration
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET_PATH = "./skin_disease_dataset_100.csv" # Expected path to dataset with folders as class labels
MODEL_SAVE_PATH = "skin_disease_model.pth"

# Define 7 classes standard in skin disease datasets like HAM10000:
# 1. Melanocytic nevi (nv)
# 2. Melanoma (mel)
# 3. Benign keratosis-like lesions (bkl)
# 4. Basal cell carcinoma (bcc)
# 5. Actinic keratoses (akiec)
# 6. Vascular lesions (vasc)
# 7. Dermatofibroma (df)
CLASSES = [
    "Melanocytic Nevi",
    "Melanoma",
    "Benign Keratosis",
    "Basal Cell Carcinoma",
    "Actinic Keratosis",
    "Vascular Lesion",
    "Dermatofibroma"
]

def get_data_transforms():
    """
    Returns data augmentation transforms for training and standard transforms for validation.
    """
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    # Calculate precision, recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    
    report = classification_report(all_labels, all_preds, target_names=CLASSES[:len(np.unique(all_labels))], zero_division=0)
    
    return epoch_loss, epoch_acc, precision, recall, f1, report

def plot_training_history(history):
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'bo-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'bo-', label='Training Acc')
    plt.plot(epochs, history['val_acc'], 'ro-', label='Validation Acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    print("Saved training history plot as 'training_history.png'")

def main():
    print(f"Using device: {DEVICE}")
    
    # Check if dataset exists
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset folder not found at '{DATASET_PATH}'.")
        print("Please structure your dataset folder as:")
        print("  dataset/train/class1/img1.jpg...")
        print("  dataset/val/class1/img1.jpg...")
        print("For simulation purposes, we will skip actual training and create dummy weights.")
        create_dummy_weights()
        return
        
    train_transform, val_transform = get_data_transforms()
    
    # Load datasets
    train_dir = os.path.join(DATASET_PATH, 'train')
    val_dir = os.path.join(DATASET_PATH, 'val')
    
    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        print("Error: train or val subdirectory not found inside dataset path.")
        return
        
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    num_classes = len(train_dataset.classes)
    print(f"Found {num_classes} classes: {train_dataset.classes}")
    
    # Initialize model
    model = get_model(num_classes=num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_val_acc = 0.0
    
    print("Starting Training...")
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc, val_prec, val_rec, val_f1, report = evaluate_model(model, val_loader, criterion, DEVICE)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] -> "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
              f"Val F1-Score: {val_f1:.4f}")
              
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Saved new best model to {MODEL_SAVE_PATH}")
            
    print("\nTraining completed! Final Evaluation:")
    print(report)
    plot_training_history(history)

def create_dummy_weights():
    """
    Creates a small file representing model weights for system deployment
    when training is not run or PyTorch is unavailable globally.
    """
    try:
        model = get_model(num_classes=7)
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"Successfully generated dummy weights file at: {MODEL_SAVE_PATH}")
    except Exception as e:
        print(f"Could not generate dummy weights: {e}")

if __name__ == '__main__':
    main()
