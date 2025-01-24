import argparse
from math import inf
import numpy as np
import torch
from torch.utils.data import DataLoader
from .dataset import DentalLoader
from models.point_net.pointnet import PointNet


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
        self.train_count += 1
        self.model.save("train")
        
        return np.mean(total_losses)


    def run(self):
        train_data_loader = self.train_files
        for epoch in range(100000):
            train_loss = self.train(epoch, train_data_loader)
            print(f"Epoch {epoch + 1}: Training Loss = {train_loss:.4f}")




def collate_fn(batch):
    output = {}

    for batch_item in batch:
        for key in batch_item.keys():
            if key not in output:
                output[key] = []
            output[key].append(batch_item[key])
    
    for output_key in output.keys():
        if output_key in ["feat", "label"]:
            output[output_key] = torch.stack(output[output_key])
    return output
def get_files(config):
    point_loader = DataLoader(
        DentalLoader(
            config["processed_data"], 
            split_with_txt_path=config["train_txt"]
        ), 
        shuffle=True,
        batch_size=config["train_batch_size"],
        collate_fn=collate_fn
    )

    return point_loader

def runner(config, model):
    train_files = get_files(config["generator"])
    print("train_files", len(train_files))
    trainer = Trainer(config=config, model=model, train_files=train_files)
    trainer.run()

def get_train_config(processed_data, train_txt):
    return {
        "generator": {
            "processed_data": f"{processed_data}",
            "train_txt": f"{train_txt}",
            "train_batch_size": 1,
        },
        "checkpoint_path": "ckpts/default_experiment",
    }

def main():
    parser = argparse.ArgumentParser(description='Training models')
    parser.add_argument('--processed_data', default="data_preprocessed", type=str, 
                        help="input data dir path.")
    parser.add_argument('--train_txt', default="train.txt", type=str, 
                        help="train cases list file path.")
    args = parser.parse_args()

    config = get_train_config(
        args.processed_data,
        args.train_txt,
    )

    model = PointNet(config)
    runner(config, model)

if __name__ == "__main__":
    main()