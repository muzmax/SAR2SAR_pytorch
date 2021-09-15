import torch.nn as nn
import torch
from .log_likelihood import log_likelihood_mean


class SAR2SARBiasLoss(torch.nn.Module):
    def __init__(self,m,M,alpha=1) -> None:
        super().__init__()
        
        self.logLikeLoss = log_likelihood_mean(m,M)
        self.lossbias = torch.nn.MSELoss()
        self.alpha = alpha
    
    def forward(self,model_output,noisy_ref,ref,mean_out):
        loss1 = self.logLikeLoss(model_output,noisy_ref)
        loss2 = self.lossbias(ref,mean_out)
        return loss1+self.alpha*loss2,loss1,self.alpha*loss2
