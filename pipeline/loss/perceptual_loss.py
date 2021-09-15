import torch
import torchvision
from .log_likelihood import log_likelihood_mean


""" Input : matrix pytorch Nb x chan x h x w 
    Output : normalized matrix, each h x w matrix is divided by the maximum"""
def normalize_output(out):
    
        shape = out.shape
        out = out.view(shape[0]*shape[1],shape[2]*shape[3])
        out_norm = out/out.max(1,keepdim=True)[0]
        out_norm = out.view(shape)
        return out_norm
    
class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self,device, resize=False):
        super(VGGPerceptualLoss, self).__init__()
        
        net = torchvision.models.vgg16(pretrained=True).features[:2].eval()
        for p in net.parameters():
            p.requires_grad = False
        self.net = net
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.nn.Parameter(torch.tensor([0.57752, 0.57752, 0.57752], device=device).view(1,3,1,1))
        self.std = torch.nn.Parameter(torch.tensor([0.04305, 0.04305, 0.04305], device=device).view(1,3,1,1))
        self.resize = resize
        self.loss = torch.nn.MSELoss()
    
    
        
                
    def forward(self, input, target):
        # Triple the gray channel for RGB input
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
            
        x = input
        x = self.net(x)
        x = normalize_output(x)
        
        y = target
        y = self.net(y)
        y = normalize_output(y)

        loss = self.loss(x, y)
        return loss
    
class SAR2SARPerceptualLoss(torch.nn.Module):
    def __init__(self,m,M,device,alpha=1,resize=True) -> None:
        super().__init__()
        self.perceptualLoss = VGGPerceptualLoss(device).to(device)
        self.logLikeLoss = log_likelihood_mean(m,M)
        self.alpha = alpha
    
    def forward(self,denoised,noisy_ref,ref):
        loss1 = self.logLikeLoss(denoised,noisy_ref)
        loss2 = self.perceptualLoss(denoised,ref)
        return loss1+self.alpha*loss2,loss1,self.alpha*loss2

class SAR2SARPerceptualLoss_only(torch.nn.Module):
    def __init__(self,m,M,device,alpha=1,resize=True) -> None:
        super().__init__()
        self.perceptualLoss = VGGPerceptualLoss(device).to(device)
        self.alpha = alpha
    
    def forward(self,denoised,noisy_ref,ref):
        loss2 = self.perceptualLoss(denoised,ref)

        return self.alpha*loss2,0,self.alpha*loss2
        
    
# loss = SAR2SARPerceptualLoss(-2,10,'cpu')
# t = torch.randn(256,256)
# ref = torch.randn(256,256)
# ref2 = torch.randn(256,256)
# print(loss(t,ref,ref2))


# class VGGPerceptualLoss2(torch.nn.Module):
#     def __init__(self,device, resize=True):
#         super(VGGPerceptualLoss, self).__init__()
#         blocks = []
#         blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
#         blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
#         blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
#         blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
#         for bl in blocks:
#             for p in bl.parameters():
#                 p.requires_grad = False
            
#         self.blocks = torch.nn.ModuleList(blocks)
#         self.transform = torch.nn.functional.interpolate
#         self.mean = torch.nn.Parameter(torch.tensor([0.57752, 0.57752, 0.57752], device=device).view(1,3,1,1))
#         self.std = torch.nn.Parameter(torch.tensor([0.04305, 0.04305, 0.04305], device=device).view(1,3,1,1))
#         self.resize = resize

#     def forward(self, input, target):
#         if input.shape[1] != 3:
#             input = input.repeat(1, 3, 1, 1)
#             target = target.repeat(1, 3, 1, 1)
#         input = (input-self.mean) / self.std
#         target = (target-self.mean) / self.std
#         if self.resize:
#             input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
#             target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
#         loss = 0.0
#         x = input
#         y = target
#         for block in self.blocks:
#             x = block(x)
#             y = block(y)
#             loss += torch.nn.functional.l1_loss(x, y)
#         return loss
