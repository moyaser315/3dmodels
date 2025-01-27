import torch
from torch.utils.data import Dataset
import os
import numpy as np
from glob import glob
from torch.utils.data import DataLoader

class DentalLoader(Dataset):
    def __init__(self, data_dir=None, train_txt=None):
        self.data_dir = data_dir
        self.files_paths = glob(os.path.join(data_dir, "*_sampled_points.npy"))

        self.files = []
        f = open(train_txt, "r")
        while True:
            line = f.readline()
            if not line:
                break
            self.files.append(line.strip())
        f.close()

        train_files = []
        for i in range(len(self.files_paths)):
            p_id = os.path.basename(self.files_paths[i]).split("_")[0]
            if p_id in self.files:
                train_files.append(self.files_paths[i])
        self.files_paths = train_files

    def __len__(self):
        return len(self.files_paths)

    def __getitem__(self, idx):
        mesh_arr = np.load(self.files_paths[idx].strip())
        output = {}
        pcd = mesh_arr.copy()[:, :6].astype("float32")
        seg_label = mesh_arr.copy()[:, 6:].astype("int")
        seg_label -= 1

        pcd = torch.from_numpy(pcd)
        pcd = pcd.permute(1, 0)
        output["feat"] = pcd
        seg_label = torch.from_numpy(seg_label)
        seg_label = seg_label.permute(1, 0)
        output["label"] = seg_label
        output["file_path"] = self.files_paths[idx]

        return output

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

def load_data(config):
    point_loader = DataLoader(
        DentalLoader(config["processed_data"], split_with_txt_path=config["train_txt"]),
        shuffle=True,
        batch_size=config["train_batch_size"],
        collate_fn=collate_fn,
    )

    return point_loader

if __name__ == "__main__":
    dt = DentalLoader("ex/")
    for batch in dt:
        for key in batch.keys():
            if type(batch[key]) == torch.Tensor:
                print(key, batch[key].shape)
            else:
                print(key, batch[key])
