import torch
import numpy as np
import argparse
from typing import List, Tuple

def compute_ece(confidences: np.ndarray, accuracies: np.ndarray, num_bins: int = 10) -> float:
    """Expected Calibration Error (ECE) calculation."""
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    binned_indices = np.digitize(confidences, bins) - 1
    
    ece = 0.0
    total_samples = len(confidences)
    
    for i in range(num_bins):
        bin_mask = binned_indices == i
        if not np.any(bin_mask):
            continue
            
        bin_confs = confidences[bin_mask]
        bin_accs = accuracies[bin_mask]
        
        bin_size = len(bin_confs)
        avg_conf = np.mean(bin_confs)
        avg_acc = np.mean(bin_accs)
        
        ece += (bin_size / total_samples) * np.abs(avg_acc - avg_conf)
        
    return ece

def evaluate_model_calibration(model, dataloader, device) -> float:
    model.eval()
    all_confs = []
    all_accs = []
    
    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch['input_ids'].to(device), batch['labels'].to(device)
            outputs = model(inputs)
            
            probs = torch.softmax(outputs.logits, dim=-1)
            confidences, predictions = torch.max(probs, dim=-1)
            
            accuracies = (predictions == labels).float()
            
            all_confs.extend(confidences.cpu().numpy().flatten())
            all_accs.extend(accuracies.cpu().numpy().flatten())
            
    return compute_ece(np.array(all_confs), np.array(all_accs))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    args = parser.parse_args()
    print(f"ECE Calibration computed for {args.model_path}")
