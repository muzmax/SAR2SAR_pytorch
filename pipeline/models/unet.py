import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

""" convolution + leaky relu """
class Conv_LR(nn.Module):
    def __init__(self, in_ch, out_ch,depth = 1):
        super().__init__()
        self.depth = depth
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3,padding=(1,1))
        self.leakyRelu  = nn.LeakyReLU(0.1)
        if depth >1:
            self.conv2 = nn.Conv2d(out_ch, out_ch, 3,padding=(1,1))
        
    def forward(self, x):
        
        if self.depth == 1:
            return self.leakyRelu(self.conv1(x))
        
        x = self.leakyRelu(self.conv1(x))
        for i in range(self.depth-1):
            x = self.leakyRelu(self.conv2(x))
        return x
    
""" One layer of the encoder"""
class Encoder(nn.Module):
    def __init__(self,in_c,out_c,depth=1):
        super().__init__()
        self.CL = Conv_LR(in_c,out_c,depth)
        self.pool = nn.MaxPool2d(2,2)
    
    def forward(self, x):
        x = self.CL(x)
        x = self.pool(x)
        
        return x

""" One layer of the decoder"""
class Decoder(nn.Module):
    def __init__(self,in_c,skip_c,out_c):
        super().__init__()
        self.CL = Conv_LR(in_c+skip_c,out_c,depth=2)
        self.up = nn.UpsamplingNearest2d(scale_factor=2)
        
    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x,skip],axis=1)
        x = self.CL(x)
        return x

""" Last layer of the decoder """    
class Out(nn.Module):
    def __init__(self,in_c,skip_c,out_c1,out_c2,out_c3=1):
        super().__init__()
        self.up = nn.UpsamplingNearest2d(scale_factor=2)
        self.CL1 = Conv_LR(in_c+skip_c,out_c1)
        self.CL2 = Conv_LR(out_c1,out_c2)
        self.C = nn.Conv2d(out_c2, out_c3, 3,padding=(1,1))
        
    def forward(self,x,skip):
        x = self.up(x)
        x = torch.cat([x,skip],axis=1)
        x = self.CL1(x)
        x = self.CL2(x)
        x = self.C(x)
        return x
    
    
""" The UNet model """    
class UNet(nn.Module):
    def __init__(self) :
        super().__init__()
        """ Encoder """
        self.e1 = Encoder(1,48,depth=2)
        self.e2 = Encoder(48,48)

        """ Bottleneck """
        self.b = Conv_LR(48,48)
        
        """ Decoder """
        self.d1 = Decoder(48,48,96)
        self.d2 = Decoder(96,48,96)
        
        """ Out """
        self.o = Out(96,1,64,32,1)
        
    def forward(self,input):
        
        s1 = self.e1(input)
        s2 = self.e2(s1)
        s3 = self.e2(s2)
        s4 = self.e2(s3)
        s = self.e2(s4)
        s = self.b(s)
        s = self.d1(s,s4)
        s = self.d2(s,s3)
        s = self.d2(s,s2)
        s = self.d2(s,s1)
        s = self.o(s,input)
        
        return input-s
  




