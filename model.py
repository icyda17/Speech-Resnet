import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum

class SerializableModule(nn.Module):
    def __init__(self):
        super().__init__()

    def save(self, filename):
        torch.save(self.state_dict(), filename)
    
    def load(self, filename):
        self.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))
    
class SpeechRes(SerializableModule):
    def __init__(self, config):
        super().__init__()
        n_labels = config['n_labels']
        n_maps = config['n_feature_maps']
        self.conv0 = nn.Conv2d(1, n_maps, (3,3), padding=(1,1), bias = False)
        if 'res_pool' in config:
            self.pool = nn.AvgPool2d(config['res_pool'])
        
        self.n_layers = config['n_layers']
        dilation = config['use_dilation']
        if dilation:
            self.convs = [nn.Conv2d(n_maps, n_maps, (3,3), padding=int(2**(i//3)), dilation=int(2**(i//3)), bias = False) for i in range(self.n_layers)]
        else:
            self.convs = [nn.Conv2d(n_maps, n_maps, (3,3), padding=1, dilation=1, bias = False) for _ in range(self.n_layers)]
        
        for i, conv in enumerate(self.convs):
            self.add_module("bn%s"%(i+1), nn.BatchNorm2d(n_maps, affine=False))
            self.add_module('conv%s'%(i+1), conv)
        
        self.output = nn.Linear(n_maps, n_labels)

    def forward(self, x):
        x = x.unsqueeze(1)
        for i in range(self.n_layers + 1):
            y = F.relu(getattr(self, "conv%s"%i)(x))
            if i == 0:
                if hasattr(self, "pool"):
                    y = self.pool(y)
                old_x = y
            if i > 0 and i %2 ==0:
                x = y + old_x
                old_x = x
            else:
                x = y
            if i > 0:
                x = getattr(self, 'bn%s'%i)(x)
        x = x.view(x.size(0), x.size(1), -1) # (batch, feats, o3)
        x = torch.mean(x, 2)
        return self.output(x)

class ConfigType(Enum):
    RES15 = "res15"
    RES26 = "res26"
    RES8 = "res8"

_configs = {
    ConfigType.RES15.value: dict(n_labels = 12, n_layers = 13, use_dilation= True, n_feature_maps= 45),
    ConfigType.RES8.value: dict(n_labels = 12, n_layers = 6, res_pool=(4,3), use_dilation= False, n_feature_maps= 45),
    ConfigType.RES26.value: dict(n_labels = 12, n_layers = 24, use_dilation= False, n_feature_maps= 45),
    ConfigType.RES15_NARROW.value: dict(n_labels=12, use_dilation=True, n_layers=13, n_feature_maps=19),
    ConfigType.RES8_NARROW.value: dict(n_labels=12, n_layers=6, n_feature_maps=19, res_pool=(4, 3), use_dilation=False),
    ConfigType.RES26_NARROW.value: dict(n_labels=12, n_layers=24, n_feature_maps=19, res_pool=(2, 2), use_dilation=False)
}
