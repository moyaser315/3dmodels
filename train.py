import argparse
import json
import numpy as np
import sys
import os
from tqdm import tqdm
from .dataset.dataset import load_data
from models.point_net.pointnet import PointNet


sys.path.append(os.getcwd())


class Trainer:
    def __init__(self, config, model, train_files, test_files=None):
        self.train_files = train_files
        self.test_files = test_files
        self.config = config
        self.model = model
        self.train_count = 0
        
        
        if "checkpoint_path" in config and config["checkpoint_path"]:
            self.model.load()
        

        self.history = {
            'train_losses': [],
            'test_losses': [],
            'epochs': []
        }

    def train(self, epoch, data_loader):
        total_losses = []

        # Create progress bar for training batches
        train_pbar = tqdm(data_loader, desc=f"Epoch {epoch+1} - Training", 
                         leave=False, ncols=100)
        
        for batch_idx, batch_item in enumerate(train_pbar):

            loss = self.model.step(batch_idx, batch_item, "train")
            total_losses.append(loss.item())

            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{np.mean(total_losses):.4f}'
            })

        self.model.scheduler.step()
        self.train_count += 1
        
        return np.mean(total_losses)

    def test(self, epoch, data_loader):

        if data_loader is None:
            return None
            
        total_losses = []
        
        # Create progress bar for testing batches
        test_pbar = tqdm(data_loader, desc=f"Epoch {epoch+1} - Testing", 
                        leave=False, ncols=100)
        
        for batch_idx, batch_item in enumerate(test_pbar):
            loss = self.model.step(batch_idx, batch_item, "test")
            total_losses.append(loss.item())
            
            test_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{np.mean(total_losses):.4f}'
            })
        
        return np.mean(total_losses)

    def save_metrics(self, epoch, train_loss, test_loss=None):
        """Save losses to JSON file"""
        self.history['epochs'].append(epoch + 1)
        self.history['train_losses'].append(train_loss)
        
        if test_loss is not None:
            self.history['test_losses'].append(test_loss)
        else:
            self.history['test_losses'].append(None)
        
        metrics_file = "/kaggle/working/training_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"Metrics saved to {metrics_file}")

    def run(self):
        train_data_loader = self.train_files
        

        epoch_pbar = tqdm(range(100000), desc="Training Progress", ncols=120)
        
        for epoch in epoch_pbar:

            train_loss = self.train(epoch, train_data_loader)
            
            test_loss = None
            if self.test_files is not None:
                test_loss = self.test(epoch, self.test_files)
                
                # Update main progress bar with metrics
                epoch_pbar.set_postfix({
                    'Train Loss': f'{train_loss:.4f}',
                    'Test Loss': f'{test_loss:.4f}'
                })
                
                print(f"Epoch {epoch + 1}: Train Loss={train_loss:.4f} | Test Loss={test_loss:.4f}")
            else:
                # Update main progress bar with train metrics only
                epoch_pbar.set_postfix({
                    'Train Loss': f'{train_loss:.4f}'
                })
                
                print(f"Epoch {epoch + 1}: Train Loss={train_loss:.4f}")
            
            # Save model and metrics every 5 epochs
            if (epoch + 1) % 5 == 0:
                self.model.save("train")
                self.save_metrics(epoch, train_loss, test_loss)
                print(f"Model and metrics saved at epoch {epoch + 1}")




def runner(config, model):
    print("="*60)
    print("INITIALIZING TRAINING SETUP")
    print("="*60)
    
    print("Loading training data...")
    train_files = load_data(config, config["train_txt"], "train")
    
    print("Loading test data...")
    test_files = load_data(config, config["test_txt"], "test")
    
    print("="*60)
    print(f"DATASET SUMMARY:")
    print(f"Training files: {len(train_files.dataset)} (expected: ~840 from ~420 IDs)")
    print(f"Test files: {len(test_files.dataset)} (expected: 360 from 180 IDs)")
    print(f"Total files: {len(train_files.dataset) + len(test_files.dataset)}")
    print(f"Batch size: {config['train_batch_size']}")
    print(f"Training batches per epoch: {len(train_files)}")
    print(f"Test batches per epoch: {len(test_files)}")
    print("="*60)
    
    # Verify the split is correct
    expected_test = 360  
    expected_train = 840  
    expected_total = 1200
    
    if len(test_files.dataset) != expected_test:
        print(f"⚠️  WARNING: Expected {expected_test} test files, got {len(test_files.dataset)}")
    if len(train_files.dataset) != expected_train:
        print(f"⚠️  WARNING: Expected {expected_train} train files, got {len(train_files.dataset)}")
    if len(train_files.dataset) + len(test_files.dataset) != expected_total:
        print(f"⚠️  WARNING: Expected {expected_total} total files, got {len(train_files.dataset) + len(test_files.dataset)}")
    else:
        print("✅ Dataset split looks correct!")
        print(f"✅ Test: 180 IDs → {len(test_files.dataset)} files")
        print(f"✅ Train: {len(train_files.dataset)//2} IDs → {len(train_files.dataset)} files")
    
    trainer = Trainer(config=config, model=model, train_files=train_files, test_files=test_files)
    trainer.run()

def main():
    config = {
        "processed_data": "/kaggle/input/offline-sampled-data/processed",
        "train_txt": "/kaggle/input/txtfiles/base_name_train_fold.txt",  
        "test_txt": "/kaggle/input/txtfiles/base_name_test_fold.txt",   
        "train_batch_size": 1,
        "checkpoint_path": "./chkpoints/pointnet.pt",
    }

    model = PointNet(config)
    runner(config, model)


if __name__ == "__main__":
    main()
