from configs.predict_config import PredictConfigBase
import os

from pipeline.datasets.datasets import test_data
from pipeline.datasets.load import load_eval_data
from pipeline.datasets.preprocessing import *

from torch.nn import DataParallel
from torchvision import transforms

from pipeline.models.unet import UNet
from pipeline.predictor.pred_unsupervised import PredictorUnsupervised




class ConfigSAR2SAR_pred(PredictConfigBase):
    def __init__(self,eval_dir,save_dir,weights_dir,norm,add_speck = False,batch_size=1,print_freq = '1'):
        
        eval = load_eval_data(eval_dir)
        if add_speck == True:
            process_eval = transforms.Compose([add_speckle(),normalization(norm[0],norm[1],norm[2]),ToTensor()])
        else:  
            process_eval = transforms.Compose([normalization(norm[0],norm[1],norm[2]),ToTensor()])

        eval_dataset = test_data(eval,process_eval)
            
        model = UNet()
        print_frequency = print_freq
        device = 'cuda'
        num_workers = 0
        model_path = weights_dir
        pred = PredictorUnsupervised
        
        
        self.norm = norm
        self.speckle = add_speck
        super().__init__(model=model,
                         dataset=eval_dataset,
                         model_save_path = model_path,
                         predictor = pred,
                         save_im_path = save_dir,
                         device=device,
                         batch_size=batch_size,
                         num_workers=num_workers,
                         print_frequency=print_frequency)
            
                
        