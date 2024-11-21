# Optic Disc Disease Classification using Multi-Attention Block with Combined CNN


class_names = ['Atrophy', 'Cupping', 'Drusen', 'Normal', 'Papilledema', 'Tilted']

# Performance Metrics for each fold
![fold_metrics_plot](https://github.com/user-attachments/assets/2c8699ca-d3d0-4f89-aebc-e5e245259810)

# CLI Run 
```bash
python opticdisc.py --batch_size 32 --learning_rate 0.0001 --epochs 50 --model AttentionCNNCombined --log_dir ./logs --checkpoint_dir ./checkpoints --device cuda
```
![CLI_run](https://github.com/user-attachments/assets/87c56bae-7359-4222-b19b-2b23a90d079b)

#Test Scores
![test_scores](https://github.com/user-attachments/assets/e41d1e4f-bec9-4fa8-96f3-79ff2ead04d4)

# Code run on Notebook
```bash
!git clone https://github.com/ogokk/optic_disc.git
cd optic_disc
pip install -r requirements.txt
```




