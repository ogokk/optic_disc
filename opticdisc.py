
import argparse
import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import KFold
from torchvision import datasets, transforms
from sklearn.metrics import matthews_corrcoef
import numpy as np
from torchvision import models
from efficientnet_pytorch import EfficientNet
import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2 as ToTensor
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F

seed = 42
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
os.environ["PYTHONHASHSEED"] = str(seed)

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

set_seed(seed)

def parse_args():
    parser = argparse.ArgumentParser(description="Deep Neural Network for Optic Disc Image Classification")
    
    # Hyperparameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate for the optimizer')
    parser.add_argument('--weight_decay_rate', type=float, default=0.0001, help='Weight decay rate for the AdamW optimizer')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs for training')
    parser.add_argument('--model', type=str, default='AttentionCNNCombined', choices=['AttentionCNNCombined'], help='Model type')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Directory for TensorBoard logs')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Directory for model checkpoints')
    parser.add_argument('--device', type=str, default=None, choices=['cpu', 'cuda'], 
                        help='Device to run the model on. "cpu" or "cuda". Default is "cuda" if a GPU is available.')

    args = parser.parse_args()
    
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return vars(args)


# Set up logging
def setup_logging(log_dir='./logs'):
    # Create log directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_filename = os.path.join(log_dir, "training.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    logging.info("Logging setup complete. Logs are saved in '%s'", log_filename)

def log_epoch_metrics(epoch, metrics):
    logging.info(f"Epoch {epoch}: Train Loss = {metrics['train_loss']:.4f}, "
                 f"Train Accuracy = {metrics['train_accuracy']:.2f}%, "
                 f"Train Dice = {metrics['train_dice']:.2f}, "
                 f"Train MCC = {metrics['train_mcc']:.2f}, "
                 f"Val Loss = {metrics['val_loss']:.4f}, "
                 f"Val Accuracy = {metrics['val_accuracy']:.2f}%,"
                 f"Val Dice = {metrics['val_dice']:.4f}, "
                 f"Val MCC = {metrics['val_mcc']:.4f} ")



albumentations_transforms = A.Compose([ A.HorizontalFlip(p=0.5),
                                        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.0, 0.1), always_apply=None, p=0.5),
                                        
                            A.OneOf([A.CLAHE(clip_limit=1.0, tile_grid_size=(1, 1), always_apply=None), A.RandomBrightnessContrast()], p=0.5),
                            A.GaussNoise(var_limit=(0, 10), mean=np.random.choice([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]), p=0.5),
                            A.Resize(height=224, width=224),
                            A.Normalize(),
                            ToTensor()
                                    ])

class Transform():
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, image):
        return self.transform(image=np.array(image))["image"]


train_dir = 'C:/Users/ProArt/Desktop/ozan/opticdisc/Train'


class AttentionCNNCombined(nn.Module):
    def __init__(self, n_class = 6, num_heads=8, hidden_dim=256):
        super(AttentionCNNCombined, self).__init__()
        
        self.resnet = models.resnet101(pretrained=True)
        self.resnet.fc = nn.Identity()  # Remove the final fully connected layer
        
        self.efficientnet = EfficientNet.from_pretrained("efficientnet-b0")
        self.efficientnet._fc = nn.Identity()

        resnet_output_dim = 2048
        efficientnet_output_dim = 1280
        
        self.attention = nn.MultiheadAttention(embed_dim=resnet_output_dim + efficientnet_output_dim, 
                                               num_heads=num_heads, dropout=0.5)
        
        self.fc = nn.Sequential(
            nn.Linear(resnet_output_dim + efficientnet_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_class)
        )

    def forward(self, x):
        resnet_out = self.resnet(x)  # Shape: (batch_size, resnet_channels)
        efficientnet_out = self.efficientnet(x)  # (batch_size, efficientnet_channels)

        # print(f"efficientnet shape: {efficientnet_out.shape}")
        # print(f"resnet50 shape: {resnet_out.shape}")

        combined_features = torch.cat((resnet_out, efficientnet_out), dim=1)  # Shape: (batch_size, combined_channels)
        combined_features = combined_features.unsqueeze(0)  
        attn_output, _ = self.attention(combined_features, combined_features, combined_features)  # Self-attention for query, key, values
        attn_output = attn_output.squeeze(0)
        
        output = self.fc(attn_output)  #(batch_size, num_classes)
        
        return output


model = AttentionCNNCombined(n_class = 6)


def dice_coefficient(pred, target, threshold=0.5):
    pred = pred > threshold
    target = target > threshold
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    return 2 * intersection / union if union > 0 else 0


def mcc(pred, target):
    return matthews_corrcoef(target.flatten(), pred.flatten())



def train_epoch(model, train_loader, criterion, optimizer, config):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    itr = 0
    accumulation_steps = 64 // len(train_loader)

    for inputs, labels in tqdm(train_loader):
        inputs, labels = inputs.to(config['device']), labels.to(config['device'])
        outputs = model(inputs).to(config['device'])
        loss = criterion(outputs, labels)
        loss = loss / accumulation_steps
        loss.backward()
        #Gradient Accumulation
        if (itr + 1 ) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        itr += 1

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_preds.append(predicted)
        all_labels.append(labels)

    accuracy = 100 * correct / total
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    
    # Calculate metrics
    dice_train = dice_coefficient(all_preds.cpu().numpy(), all_labels.cpu().numpy())
    mcc_train = mcc(all_preds.cpu().numpy(), all_labels.cpu().numpy())

    return running_loss / len(train_loader), accuracy, dice_train, mcc_train 



def validate_epoch(model, val_loader, criterion, config):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader):
            inputs = inputs.to(config['device'])
            labels = labels.to(config['device'])
            
            outputs = model(inputs).to(config['device'])  
            outputs = outputs.view(outputs.size(0), -1)
            
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.append(predicted)
            all_labels.append(labels)
    
    accuracy = 100 * correct / total
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    
    dice_val = dice_coefficient(all_preds.cpu().numpy(), all_labels.cpu().numpy())
    mcc_val = mcc(all_preds.cpu().numpy(), all_labels.cpu().numpy())
    
    return running_loss / len(val_loader), accuracy, dice_val, mcc_val

fold_metrics = []

def main():
    config = parse_args()
    setup_logging(config['log_dir'])
    train_dataset = ImageFolder(train_dir, transform=Transform(albumentations_transforms))

    k = 5  # k fold cross-validation
    kfold = KFold(n_splits=k, shuffle=True, random_state=5)
    
    checkpoint_dir = './checkpoints'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        
    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_dataset)):
        model = AttentionCNNCombined(n_class=6).to(config['device'])
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay = config['weight_decay_rate'] )
        scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=3, verbose=True) 
        start = time.strftime("%H:%M:%S")
        print(f"----------- Fold: {fold + 1}  --- time: {start} -----------")
        logging.info(f"Training fold {fold+1}/{k}...")
        
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)
        
        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], sampler=train_subsampler)
        val_loader = DataLoader(train_dataset, batch_size=config["batch_size"], sampler=val_subsampler)
        
        best_val_acc = -float('inf')
        best_epoch = 0
        best_checkpoint_path = ""


        for epoch in range(config["epochs"]):
            start_time = time.strftime("%H:%M:%S")
            print(f"----------- Epoch {epoch + 1}/{config['epochs']} --- time: {start_time} -----------")
            logging.info(f"Training epoch {epoch+1}/{config['epochs']}...")
    
            # Training phase
            train_loss, train_acc, dice_train, mcc_train = train_epoch(model, train_loader, criterion, optimizer, config)
            print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.2f}% | "
                  f"Train Dice: {dice_train:.4f} | "
                  f"Train MCC: {mcc_train:.4f}")

            logging.info(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.2f}% | "
                                     f"Train Dice: {dice_train:.4f} | "
                                     f"Train MCC: {mcc_train:.4f}")

    
            # Validation phase
            val_loss, val_acc, val_dice, val_mcc = validate_epoch(model, val_loader, criterion, config)
            print(f"Val Loss: {val_loss:.4f} | Val Accuracy: {val_acc:.2f}% | "
                  f"Val Dice: {val_dice:.4f} | "
                  f"Val MCC: {val_mcc:.4f}")
            
            logging.info(f"Val Loss: {val_loss:.4f} | Val Accuracy: {val_acc:.2f}% | "
             f"Val Dice: {val_dice:.4f} | "
             f"Val MCC: {val_mcc:.4f}")

            scheduler.step(val_loss)
            
            # Log epoch metrics
            log_epoch_metrics(epoch + 1, {
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "train_dice": dice_train,
                "train_mcc": mcc_train,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_dice": val_dice,
                "val_mcc": val_mcc,
            })

            print("\n}")


            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1 
                best_checkpoint_path = os.path.join(config['checkpoint_dir'], f"best_model_fold_{fold + 1}_epoch_{best_epoch}.pth")
            
                torch.save(model.state_dict(), best_checkpoint_path)
                print(f"Best model checkpoint saved for fold {fold + 1}, epoch {best_epoch} to {best_checkpoint_path}")
    
            fold_metrics.append({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "train_dice": dice_train,
                "train_mcc": mcc_train,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_dice": val_dice,
                "val_mcc": val_mcc
            })
            
        # Print summary of training and validation metrics after all epochs
        for metrics in fold_metrics:
            print(f"Epoch {metrics['epoch']} - Train Loss: {metrics['train_loss']:.2f} | "
                  f"Train Accuracy: {metrics['train_accuracy']:.2f}% | "
                  f"Train Dice: {metrics['train_dice']:.2f} | "
                  f"Train MCC: {metrics['train_mcc']:.2f} | "
                  f"Val Loss: {metrics['val_loss']:.2f}% | "
                  f"Val Accuracy: {metrics['val_accuracy']:.2f}% | "
                  f"Val Dice: {metrics['val_dice']:.4f} | "
                  f"Val MCC: {metrics['val_mcc']:.4f}")



if __name__ == "__main__":
    main()


#------------------------------------------------------------------------------
## PLOT the Metrics: Train and Val loss, Accuracies, Dices and MCC's
#------------------------------------------------------------------------------

import matplotlib.pyplot as plt

num_folds = 5
epochs_per_fold = 50

# Metrics for plotting
epochs = [metrics['epoch'] for metrics in fold_metrics]
train_loss = [metrics['train_loss'] for metrics in fold_metrics]
train_accuracy = [metrics['train_accuracy'] for metrics in fold_metrics]
train_dice = [metrics['train_dice'] for metrics in fold_metrics]
train_mcc = [metrics['train_mcc'] for metrics in fold_metrics]
val_loss = [metrics['val_loss'] for metrics in fold_metrics]
val_accuracy = [metrics['val_accuracy'] for metrics in fold_metrics]
val_dice = [metrics['val_dice'] for metrics in fold_metrics]
val_mcc = [metrics['val_mcc'] for metrics in fold_metrics]

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.flatten()
metrics = [
    ('Train - Val Loss', train_loss, val_loss),
    ('Train - Val Accuracy', train_accuracy, val_accuracy),
    ('Train - Val Dice Coefficients', train_dice, val_dice),
    ('Train - Val MCC', train_mcc, val_mcc),
]

for i, (metric_name, train_data, val_data) in enumerate(metrics):
    ax = axes[i]
    
    for fold in range(num_folds):
        start_idx = fold * epochs_per_fold
        end_idx = (fold + 1) * epochs_per_fold

        fold_epochs = epochs[start_idx:end_idx]
        fold_train_data = train_data[start_idx:end_idx]
        fold_val_data = val_data[start_idx:end_idx]

        # Plot train and validation data for this fold
        ax.plot(fold_epochs, fold_train_data, label=f"Fold {fold + 1} Train", linestyle='-', marker='o')
        ax.plot(fold_epochs, fold_val_data, label=f"Fold {fold + 1} Val", linestyle='--', marker='x')

    
    ax.set_title(f'{metric_name} for Each Fold')
    ax.set_xlabel('Epochs #')
    ax.set_ylabel(metric_name)
    ax.grid(True)
    ax.legend()


plt.tight_layout()
fig.savefig('fold_metrics_plot.png', dpi=600)
plt.show()



#-----------------------TEST Phase--------------------------------------------

from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score, confusion_matrix


albumentations_transforms = A.Compose([
    A.Resize(height=224, width=224),
    A.Normalize(),
    ToTensor()
])

class Transform():
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, image):
        return self.transform(image=np.array(image))["image"]

# Directory path for Test dataset
test_dir = 'C:/Users/ProArt/Desktop/ozan/opticdisc/Test'
batch_size = 16

def test_model(model, test_loader, device):
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.append(predicted.cpu())
            all_labels.append(labels.cpu())
    
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    
    accuracy = accuracy_score(all_labels, all_preds)
    mcc = matthews_corrcoef(all_labels, all_preds)
    dice = f1_score(all_labels, all_preds, average="macro")
    class_names = ['Atrophy', 'Cupping', 'Drusen', 'Normal', 'Papilledema', 'Tilted']
    all_preds_str = [class_names[pred] for pred in all_preds.numpy()]
    all_labels_str = [class_names[label] for label in all_labels.numpy()]
    
    confmat = confusion_matrix(all_labels_str, all_preds_str, labels=class_names)
    
    return accuracy, mcc, dice, confmat

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    test_dataset = ImageFolder(test_dir, transform=Transform(albumentations_transforms))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    best_epoch_index = max(range(len(fold_metrics)), key=lambda i: (fold_metrics[i]['val_accuracy'], -fold_metrics[i]['val_loss']))
    best_model_metrics = fold_metrics[best_epoch_index]
    fold_number = int(np.ceil(best_epoch_index/50))
    best_epoch = best_model_metrics['epoch']
    checkpoint_path = f"C:/Users/ProArt/Desktop/ozan/opticdisc/checkpoints/best_model_fold_{fold_number}_epoch_{best_epoch}.pth"

    
    model.load_state_dict(torch.load(checkpoint_path))
    model.to(device)
    model.eval()
    
    accuracy, mcc, dice, confmat = test_model(model, test_loader, device)
    print("\n")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print(f"Test MCC: {mcc:.4f}")
    print(f"Test Dice: {dice:.4f}")
    print(f"Confusion Matrix: \n{confmat}")

if __name__ == "__main__":
    main()



