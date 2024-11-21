# Optic Disc Classification using Multi-Attention Block with Combined CNN


class_names = ['Atrophy', 'Cupping', 'Drusen', 'Normal', 'Papilledema', 'Tilted']

# Performance Metrics for each fold
![fold_metrics_plot](https://github.com/user-attachments/assets/2c8699ca-d3d0-4f89-aebc-e5e245259810)

# Run from CLI 
```bash
python opticdisc.py --batch_size 32 --learning_rate 0.0001 --epochs 50 --model AttentionCNNCombined --log_dir ./logs --checkpoint_dir ./checkpoints --device cuda
```
![CLI_run](https://github.com/user-attachments/assets/87c56bae-7359-4222-b19b-2b23a90d079b)

# Test Scores
Ordered class names for confusion matrix : 'Atrophy', 'Cupping', 'Drusen', 'Normal', 'Papilledema', 'Tilted'

![test_scores](https://github.com/user-attachments/assets/e41d1e4f-bec9-4fa8-96f3-79ff2ead04d4)

# Code run on Notebook
```bash
!git clone https://github.com/ogokk/optic_disc.git
cd optic_disc
pip install -r requirements.txt
```
# Best model selection 
```python
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

test_dataset = ImageFolder(test_dir, transform=Transform(albumentations_transforms))
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
best_epoch_index = max(range(len(fold_metrics)), key=lambda i: (fold_metrics[i]['val_accuracy'], -fold_metrics[i]['val_loss']))
best_model_metrics = fold_metrics[best_epoch_index]
fold_number = int(np.ceil(best_epoch_index/50))
best_epoch = best_model_metrics['epoch']
checkpoint_path = f"C:/Users/ProArt/Desktop/ozan/opticdisc/checkpoints/best_model_fold_{fold_number}_epoch_{best_epoch}.pth"

# Load the model
model.load_state_dict(torch.load(checkpoint_path))
model.to(device)
model.eval()
# Evaluate the model
accuracy, mcc, dice, confmat = test_model(model, test_loader, device)
print("\n")
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"Test MCC: {mcc:.4f}")
print(f"Test Dice: {dice:.4f}")
print(f"Confusion Matrix: \n{confmat}")
```



