import time
from typing import Iterable

import torch
import torch.nn as nn

import numpy as np

from ..logger import LOGGER
from ..utils import move_to_device, load_model, save_im, tresh_im
from ..datasets.preprocessing import denormalization

import os

save_tresh = 430 # set variable to None for automatic tresholding

class PredictorUnsupervised:
    def __init__(
            self,
            model: nn.Module,
            data_loader: Iterable,
            print_frequency: None or int,
            device: str,
            model_save_path: str,
            speckle,
            norm,
            save_im_path:str) -> None:

        self.model = model.to(device)
        self.data_loader = data_loader
        self.print_frequency = print_frequency
        self.device = device
        self.model_save_path = model_save_path
        self.speckle = speckle
        self.norm = norm
        self.save_im_path = save_im_path

    def predict_step(self, input_data: torch.Tensor,name: str, step_id,denorm):
        patch_size = 256
        stride = 32        
#         if self.speckle == True:
#             input_np = denorm(input_data.cpu().data.numpy())
#             save_im(tresh_im(input_np,treshold=save_tresh),'{}/noisy{}.png'.format(self.save_im_path,step_id))
#             np.save('{}/noisy{}.npy'.format(self.save_im_path,step_id),input_np)
        
        # Save noisy image to png and npy format   
        input_np = denorm(input_data.cpu().data.numpy())
        save_im(tresh_im(input_np,treshold=save_tresh),'{}/noisy_{}.png'.format(self.save_im_path,name[0]))
        np.save('{}/noisy_{}.npy'.format(self.save_im_path,name[0]),input_np)
         
        input_data = move_to_device(input_data, device=self.device)
        (un,c,h,w) = input_data.shape
        result = torch.zeros(h,w,device=self.device)
        count = torch.zeros(h,w,device=self.device)
        
        if h == patch_size:
            x_range = list(np.array([0]))
        else:
            x_range = list(range(0,h-patch_size,stride))
            if (x_range[-1]+patch_size)<h : x_range.extend(range(h-patch_size,h-patch_size+1))
        
        if w == patch_size:
            y_range = list(np.array([0]))
        else:
            y_range = list(range(0,w-patch_size,stride))
            if (y_range[-1]+patch_size)<w : y_range.extend(range(w-patch_size,w-patch_size+1))
            
        for x in x_range:
            for y in y_range:
                clean = self.model(input_data[:,:,x:x+patch_size,y:y+patch_size])
                result[x:x+patch_size,y:y+patch_size] += torch.squeeze(clean)
                count[x:x+patch_size,y:y+patch_size] += torch.ones(patch_size,patch_size,device=self.device)
        result = torch.div(result,count)
        
        # Save denoised image to png and npy format   
        result = denorm(result.cpu().data.numpy())
        save_im(tresh_im(result,treshold=save_tresh),'{}/denoised_{}.png'.format(self.save_im_path,name[0]))
        np.save('{}/denoised_{}.npy'.format(self.save_im_path,name[0]),result)
        

    def log_predict_step(self, step_id: int, predict_time: float):
        if self.print_frequency is None or step_id % self.print_frequency == 0:
            LOGGER.info("[{} s] Predict step {}".format(predict_time, step_id))
            return True

        return False

    def log_predict_completed(self, predict_time: float):
        LOGGER.info("[{} s] Predict is completed".format(predict_time))
        return True

    """ Load latest model in folder self.model_save_path """
    def load_last_model(self):
        if os.path.exists(self.model_save_path):
            phases = ['C','B','A']
            for phase in phases:
                epochs = filter(lambda file: file.startswith("{}_epoch_".format(phase)), os.listdir(self.model_save_path))
                epochs = map(lambda file: int(file[file.find("h_")+2:]), epochs)
                epochs = list(epochs)

                if epochs:
                    last_model_path = os.path.join(self.model_save_path, "{}_epoch_{}".format(phase,max(epochs)))
                    load_model(self.model, last_model_path)
                    LOGGER.info("Model found with phase {} at epoch {}...".format(phase,max(epochs)))
                    return

        LOGGER.info("Model not found in {}. Starting to train a model from scratch...".format(self.model_save_path))

    def run(self):
        self.load_last_model()
        self.model.eval()

        denorm = denormalization(self.norm[0],self.norm[1],self.norm[2])
        step_count = 0
        start_time = time.time()

        with torch.no_grad():
            for step_id, (input_data,name) in enumerate(self.data_loader):
                self.predict_step(input_data,name,step_id,denorm)

                step_count += 1
                predict_time = time.time() - start_time
                self.log_predict_step(step_id, predict_time)

        predict_time = time.time() - start_time
        self.log_predict_completed(predict_time)
        return predict_time
