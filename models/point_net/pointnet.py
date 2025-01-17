import torch
import torch.nn as nn
import torch.nn.functional as F
from .pointnet_encoder import PointNetEncoder

class PointNet(nn.Module):
    def __init__(self):
        super(PointNet, self).__init__()

        # Module components
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
        
        # Initialize model state
        self.train()
        self.cuda()
        
        # Setup optimizer 
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=1e-3,
            weight_decay=1.0e-4
        )


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

    def load(self,path):
        self.load_state_dict(torch.load(path))
        print("Checkpoint loaded successfully.")

    def save(self, phase):
        save_path = f'/kaggle/working/pointnet_{phase}.pt'
        torch.save(self.state_dict(), save_path)
        print(f"Checkpoint saved as {save_path}.")




    

if __name__ == '__main__':
    model = PointNet()
    xyz = torch.rand(12, 3, 2048)
    (model(xyz))