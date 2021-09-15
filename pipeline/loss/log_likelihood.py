import torch.nn as nn
import torch

class log_likelihood(nn.Module):
    
    def __init__(self,m,M) -> None:
        super().__init__()
        assert isinstance(m,(float,int))
        assert isinstance(M,(float,int))
        self.m_ = m
        self.M_ = M
        
    def forward(self,denoised,ref):
        
        batch_size = denoised.shape[0]
        coef1 = torch.mul(torch.sub(denoised,ref),2*(self.M_-self.m_))
        coef2 = torch.exp(torch.mul(torch.sub(ref,denoised),2*(self.M_-self.m_)))
        loss = torch.div(torch.sum(torch.add(coef1,coef2)),batch_size)
        return loss
    
class log_likelihood_mean(nn.Module):
    
    def __init__(self,m,M) -> None:
        super().__init__()
        assert isinstance(m,(float,int))
        assert isinstance(M,(float,int))
        self.m_ = m
        self.M_ = M
        
    def forward(self,denoised,ref):
        
        batch_size = denoised.shape[0]
        coef1 = torch.mul(torch.sub(denoised,ref),2*(self.M_-self.m_))
        coef2 = torch.exp(torch.mul(torch.sub(ref,denoised),2*(self.M_-self.m_)))
        loss = torch.mean(torch.add(coef1,coef2))
        return loss
 

