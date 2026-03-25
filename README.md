# FAN-X: Cross-Branch Feature Fusion Attention Network 
<p align="center">
  <a href="https://doi.org/10.1016/j.ajoint.2026.100249">
    <img src="https://img.shields.io/badge/Paper-View-blue?style=for-the-badge">
  </a>
</p>

![Figure 1 (1)](https://github.com/user-attachments/assets/1bcb8c7a-b281-403e-9aa9-61cee0f6792d)


# Run from CLI 
```bash
python opticdisc.py --batch_size 32 --learning_rate 0.0001 --epochs 50 --model AttentionCNNCombined --log_dir ./logs --checkpoint_dir ./checkpoints --device cuda
```
![CLI_run](https://github.com/user-attachments/assets/87c56bae-7359-4222-b19b-2b23a90d079b)


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


model.load_state_dict(torch.load(checkpoint_path))
model.to(device)
model.eval()

accuracy, mcc, dice, confmat = test_model(model, test_loader, device)
print("\n")
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"Test MCC: {mcc:.4f}")
print(f"Test Dice: {dice:.4f}")
print(f"Confusion Matrix: \n{confmat}")
```


## Citation

```bibtex
@article{durmazengin2026,
  title={Evaluation of Automated Machine Learning Platforms Against an Expert-Designed Deep Learning Model for Optic Disc Abnormality Detection},
  author={Durmaz Engin, Ceren and Gokkan, Mahmut Ozan and Ozizmirliler, Denizcan and Besenk, Ufuk and Selver, Mustafa Alper and Soylev Bajin, Meltem and Grzybowski, Andrzej},
  journal={American Journal of Ophthalmology International},
  year={2026},
  url={https://doi.org/10.1016/j.ajoint.2026.100249},
}
