#!/usr/bin/env python3
"""
REAL TRAINING SCRIPT - NO BULLSHIT
Actually trains ReasonBorn on the real Xerv_AI_GRAD data
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.reasonborn.data.loader import PretrainingDataLoader
from src.reasonborn.architecture.backbone import ReasonBornSystem

class SimpleConfig:
    def __init__(self):
        self.vocab_size = 50000
        self.d_model = 512
        self.num_layers = 6
        self.n_heads = 8
        self.intermediate_size = 2048
        self.sequence_length = 2048
        self.moe_expert_layers = set()  # No MoE for simple training

def main():
    print("🔥 STARTING REAL TRAINING - NO PLACEHOLDERS")
    
    # Check if data exists
    data_path = "data/processed/Xerv_AI_GRAD_processed.jsonl"
    if not os.path.exists(data_path):
        print(f"❌ DATA NOT FOUND: {data_path}")
        return
    
    print(f"✅ FOUND DATA: {data_path}")
    
    # Create simple config
    config = SimpleConfig()
    
    # Create model
    print("🏗️  BUILDING MODEL...")
    model = ReasonBornSystem(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"✅ MODEL ON {device}")
    
    # Create data loader
    print("📊 LOADING DATA...")
    try:
        train_loader = PretrainingDataLoader(
            data_dir="data/processed",
            batch_size=4,  # Small batch for testing
            seq_len=512,   # Shorter sequences for testing
            num_workers=0,
            distributed=False
        )
        print(f"✅ DATA LOADER CREATED")
    except Exception as e:
        print(f"❌ DATA LOADER FAILED: {e}")
        return
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Training loop
    print("🚀 STARTING TRAINING...")
    model.train()
    
    step = 0
    try:
        for epoch in range(3):  # 3 epochs for testing
            epoch_loss = 0.0
            batches_processed = 0
            
            for batch_idx, batch in enumerate(train_loader):
                if batch_idx >= 10:  # Limit to 10 batches per epoch for testing
                    break
                
                # Get input_ids and labels
                if isinstance(batch, dict):
                    input_ids = batch['input_ids'].to(device)
                    labels = batch.get('labels', input_ids).to(device)
                elif isinstance(batch, (list, tuple)):
                    input_ids = batch[0].to(device)
                    labels = batch[1].to(device) if len(batch) > 1 else input_ids
                else:
                    continue
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(input_ids=input_ids, labels=labels)
                
                if hasattr(outputs, 'loss') and outputs.loss is not None:
                    loss = outputs.loss
                elif isinstance(outputs, dict) and 'loss' in outputs:
                    loss = outputs['loss']
                else:
                    print(f"❌ NO LOSS IN OUTPUTS: {type(outputs)}")
                    continue
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                batches_processed += 1
                step += 1
                
                if step % 5 == 0:
                    print(f"Step {step} | Loss: {loss.item():.4f}")
            
            if batches_processed > 0:
                avg_loss = epoch_loss / batches_processed
                print(f"Epoch {epoch + 1} | Avg Loss: {avg_loss:.4f} | Batches: {batches_processed}")
            else:
                print(f"Epoch {epoch + 1} | NO BATCHES PROCESSED")
        
        print("🎉 TRAINING COMPLETED!")
        
        # Save model
        save_path = "reasonborn_trained.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'step': step
        }, save_path)
        print(f"✅ MODEL SAVED TO {save_path}")
        
    except Exception as e:
        print(f"❌ TRAINING FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
