import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class NCD(nn.Module):
    def __init__(self, mode='rc'):
        super(NCD, self).__init__()
        self.mode = mode
        net = models.resnet18(pretrained=True)        

        self.feature = nn.Sequential(*list(net.children())[0:-2]) 
    
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(net.fc.in_features, 1))     
        
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    # input x: [b, 10, 3, h, w]
    def feat(self, x):
        b = x.shape[0]

        x = x.view((b*10, -1) + x.shape[3:])                               # [b*10, 3, h, w] 
        x = self.feature(x)
        x = self.avgpool(x)
        
        x = x.view(b, 10, -1)
        x = x - 0.5 * (x[:, 0:1] + x[:, 1:2])                              # [b, 10, 512]

        return x 


    # input shape b*16*224*224
    def forward(self, x):        
        b = x.shape[0]

        # images of the choices                              
        choices = x[:, 8:].unsqueeze(dim=2)                                # [b, 8, 1, h, w]   
        
        # images of the rows
        row1 = x[:, 0:3].unsqueeze(1)                                      # [b, 1, 3, h, w]  
        row2 = x[:, 3:6].unsqueeze(1)                                      # [b, 1, 3, h, w]     
    
        row3_p = x[:, 6:8].unsqueeze(dim=1).repeat(1, 8, 1, 1, 1)          # [b, 8, 2, h, w]
        row3 = torch.cat((row3_p, choices), dim=2)                         # [b, 8, 3, h, w]

        rows = torch.cat((row1, row2, row3), dim=1)                        # [b, 10, 3, h, w] 
    
        if self.mode == 'r':
            x = self.feat(rows)

        elif self.mode == 'rc':

            # images of the columns
            col1 = x[:, 0:8:3].unsqueeze(1)                                # [b, 1, 3, h, w]  
            col2 = x[:, 1:8:3].unsqueeze(1)                                # [b, 1, 3, h, w] 
    
            col3_p = x[:, 2:8:3].unsqueeze(dim=1).repeat(1, 8, 1, 1, 1)    # [b, 8, 2, h, w]
            col3 = torch.cat((col3_p, choices), dim=2)                     # [b, 8, 3, h, w]    

            cols = torch.cat((col1, col2, col3), dim=1)                    # [b, 10, 3, h, w]        
                     
            x = self.feat(rows) + self.feat(cols)

        x = self.fc(x.view(b*10, -1))
       
        return x.view(b, 10)  
 
