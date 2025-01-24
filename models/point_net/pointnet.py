import torch
import torch.nn as nn
import torch.nn.functional as F
from .pointnet_encoder import PointNetEncoder

class PointNet(nn.Module):
    def __init__(self, config):
        super(PointNet, self).__init__()
        self.config = config
        

        self.k = 17
        scale = 2
        self.feat = PointNetEncoder(global_feat=False, feature_transform=True, channel=6, scale=scale)
        self.conv1 = torch.nn.Conv1d(1088*scale, 512*scale, 1)
        self.conv2 = torch.nn.Conv1d(512*scale, 256*scale, 1)
        self.conv3 = torch.nn.Conv1d(256*scale, 128*scale, 1)
        self.conv4 = torch.nn.Conv1d(128*scale, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512*scale)
        self.bn2 = nn.BatchNorm1d(256*scale)
        self.bn3 = nn.BatchNorm1d(128*scale)
        

        self.train()
        self.cuda()
        

        self.optimizer = torch.optim.Adam(
            self.parameters()
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=40,eta_min=1e-5)

    def forward(self, x_in):
        x = x_in[0]
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2,1).contiguous()
        x = F.log_softmax(x.view(-1,self.k), dim=-1)
        x = x.view(batchsize, n_pts, self.k).permute(0,2,1)
        
        return {"cls_pred": x, "trans_feat": trans_feat}

    def _set_model(self, phase):
        if phase == "train":
            self.train()
        elif phase in ["val", "test"]:
            self.eval()

    def load(self):
        self.load_state_dict(torch.load(self.config["checkpoint_path"]))
        print("Checkpoint loaded successfully.")

    def save(self, phase):
        save_path = f'/kaggle/working/pointnet_{phase}.pt'
        torch.save(self.state_dict(), save_path)
        print(f"Checkpoint saved as {save_path}.")

    def get_loss(self, gt_seg_label_1, sem_1):
        tooth_class_loss_1 = self.tooth_class_loss(sem_1, gt_seg_label_1, 17)
        return {
            "tooth_class_loss_1": (tooth_class_loss_1, 1),
        }

    def step(self, batch_idx, batch_item, phase):
        self._set_model(phase)

        points = batch_item["feat"].cuda()
        seg_label = batch_item["gt_seg_label"].cuda()
        
        inputs = [points, seg_label]
        
        if phase == "train":
            output = self(inputs)
        else:
            with torch.no_grad():
                output = self(inputs)
        
        # Directly compute loss
        loss = F.cross_entropy(output["cls_pred"].permute(0,2,1), seg_label.squeeze(1))
        
        if phase == "train":
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss



    

if __name__ == '__main__':
    model = PointNet()
    xyz = torch.rand(12, 3, 2048)
    (model(xyz))