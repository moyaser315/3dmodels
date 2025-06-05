import argparse
import numpy as np
import sys
import os
from .dataset.dataset import load_data
from models.point_net.pointnet import PointNet

sys.path.append(os.getcwd())


class Trainer:
    def __init__(self, config, model, train_files):
        self.train_files = train_files
        self.config = config
        self.model = model
        self.train_count = 0

    def train(self, epoch, data_loader):
        total_losses = []

        for batch_idx, batch_item in enumerate(data_loader):
            loss = self.model.step(batch_idx, batch_item, "train")
            total_losses.append(loss.item())

        self.model.scheduler.step()

        if self.train_count % 5 == 0:
            self.model.save("train")
        self.train_count += 1

        return np.mean(total_losses)

    def run(self):
        train_data_loader = self.train_files
        for epoch in range(100000):
            train_loss = self.train(epoch, train_data_loader)
            print(f"Epoch {epoch + 1}: Training Loss = {train_loss:.4f}")



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
    parser = argparse.ArgumentParser(description="Training models")
    parser.add_argument(
        "--processed_data",
        default="data_preprocessed",
        type=str,
        help="input data dir path.",
    )
    parser.add_argument(
        "--train_txt", default="train.txt", type=str, help="split txt file path."
    )
    args = parser.parse_args()

    config = {
        "processed_data": f"{args.processed_data}",
        "train_txt": f"{args.train_txt}",
        "train_batch_size": 1,
        "checkpoint_path": "./chkpoints/pointnet.pt",
    }

    model = PointNet(config)
    runner(config, model)


def test():
    config = {
        "processed_data": "/kaggle/working/",
        "train_txt": f"/kaggle/working/",
        "train_batch_size": 1,
        "checkpoint_path": "./chkpoints/pointnet.pt",
    }

    model = PointNet(config)
    runner(config, model)


if __name__ == "__main__":
    main()
