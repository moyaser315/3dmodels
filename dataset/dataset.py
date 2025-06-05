import torch
from torch.utils.data import Dataset
import os
import numpy as np
from glob import glob
from torch.utils.data import DataLoader


class DentalLoader(Dataset):
    def __init__(self, data_dir=None, train_txt=None):
        self.data_dir = data_dir
        self.train_txt = train_txt
        print("Loading file paths...")
        
        # Get all available files
        all_files = glob(os.path.join(data_dir, "*_sampled_points.npy"))
        print(f"Found {len(all_files)} total files")
        
        # If train_txt is provided, filter files based on it
        if train_txt and os.path.exists(train_txt):
            print(f"Loading ID list from {train_txt}")
            with open(train_txt, 'r') as f:
                ids = [line.strip() for line in f.readlines() if line.strip()]
            
            print(f"Found {len(ids)} IDs in txt file")
            
            # For each ID, find both upper and lower files
            self.files_paths = []
            missing_files = []
            
            for file_id in tqdm(ids, desc="Matching files", leave=False):
                # Look for both upper and lower files for this ID
                upper_file = None
                lower_file = None
                
                for file_path in all_files:
                    file_name = os.path.basename(file_path)
                    if file_id in file_name:
                        if 'upper' in file_name:
                            upper_file = file_path
                        elif 'lower' in file_name:
                            lower_file = file_path
                
                # Add found files
                if upper_file:
                    self.files_paths.append(upper_file)
                else:
                    missing_files.append(f"{file_id}_upper")
                    
                if lower_file:
                    self.files_paths.append(lower_file)
                else:
                    missing_files.append(f"{file_id}_lower")
            
            if missing_files:
                print(f"Warning: Could not find {len(missing_files)} files:")
                for missing in missing_files[:5]:  # Show first 5 missing files
                    print(f"  - {missing}")
                if len(missing_files) > 5:
                    print(f"  ... and {len(missing_files) - 5} more")
            
            print(f"Loaded {len(self.files_paths)} files from {len(ids)} IDs (expected: {len(ids) * 2})")
        else:
            # If no txt file provided, use all files
            self.files_paths = all_files
            print(f"Using all {len(self.files_paths)} files")
        
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


def load_data(config, txt_file, data_type="train"):
    print(f"Setting up {data_type} data loader from {txt_file}...")
    
    dataset = DentalLoader(config["processed_data"], txt_file)
    
    # Show dataset info
    print(f"{data_type.capitalize()} dataset: {len(dataset)} files")
    
    point_loader = DataLoader(
        dataset,
        shuffle=True if data_type == "train" else False,  # Only shuffle training data
        batch_size=config["train_batch_size"],
        collate_fn=collate_fn,
        num_workers=4,  # Add parallel loading for faster file reading
        pin_memory=True  # Speed up GPU transfer
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
