from pipeline.datasets.preprocessing import denormalization, add_speckle_pytorch
import time
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from ..core import PipelineError
from ..logger import LOGGER
from ..storage.state import StateStorageBase
from ..utils import move_to_device, save_model, load_model, save_im,tresh_im


import os

save_tresh = 430 # set variable to None for automatic tresholding
mean_nb = 5 # number of sample to average the output

class TrainerSupervisedBias:
    def __init__(
            self,
            model: nn.Module,
            train_data_loader: Iterable,
            eval_data_loader: Iterable,
            epoch_count: int,
            optimizer: Optimizer,
            scheduler,
            loss: nn.Module,
            print_frequency: None or int,
            device: str,
            model_save_path: str,
            state_storage: StateStorageBase,
            norm,
            phase: str) -> None:

        self.model = model.to(device)
        self.train_data_loader = train_data_loader
        self.eval_data_loader = eval_data_loader
        self.epoch_count = epoch_count
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss = loss
        self.print_frequency = print_frequency
        self.device = device
        self.model_save_path = model_save_path
        self.state_storage = state_storage
        self.norm = norm
        self.phase = phase
    
    
    def train_step(self,ref: torch.Tensor,speck):
        
        mean_out = torch.empty(ref.shape,device=self.device)
        
        for i in range(mean_nb):
            input_data = speck(ref)
            input_data = move_to_device(input_data, device=self.device)
            mean_out += self.model(input_data)
           
        mean_out /= mean_nb
        
        input_data = speck(ref)
        input_data = move_to_device(input_data, device=self.device)
        
        noisy_ref = speck(ref)
        noisy_ref = move_to_device(noisy_ref, device=self.device)
        
        ref = move_to_device(ref, device=self.device)
        
        model_output = self.model(input_data)

        self.optimizer.zero_grad()
        loss,loglike_loss,bias_loss = self.loss(model_output,noisy_ref,ref,mean_out)
        loss.backward()

        self.optimizer.step(closure=None)
        
        return loss.cpu().data.numpy(),loglike_loss.cpu().data.numpy(),bias_loss.cpu().data.numpy()

    def predict_step(self, input_data: torch.Tensor, name:str, epoch_id,step_id,denorm):
        patch_size = 256
        stride = 32
        
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
        result = denorm(result.cpu().data.numpy())
        save_im(tresh_im(result,treshold=save_tresh),'./data/sample/{}_e{}.png'.format(name[0],epoch_id))
        np.save('./data/sample/{}_e{}.npy'.format(name[0],epoch_id),result)
        

    def log_train_step(self, epoch_id: int, step_id: int, epoch_time: float, loss: float, bias_loss: float, loglike_loss: float):
        if self.print_frequency is None or step_id % self.print_frequency == 0:
            LOGGER.info("[{} s] Epoch {}. Train step {}. Loss {}. Loss log likelihood {}. Loss bias {}".format(
                epoch_time, epoch_id, step_id, loss,loglike_loss,bias_loss))
            return True

        return False

    def log_evaluation_step(self, epoch_id: int, step_id: int, epoch_time: float):
        if self.print_frequency is None or step_id % self.print_frequency == 0:
            LOGGER.info("[{} s] Epoch {}. Evaluation step {}".format(
                epoch_time, epoch_id, step_id))

            return True

        return False

    def log_train_epoch(self, epoch_id: int, epoch_time: float):
        LOGGER.info("Training Epoch {} has completed. Time: {}".format(
            epoch_id, epoch_time))
        return True

    def log_evaluation_epoch(self, epoch_id: int, epoch_time: float):
        LOGGER.info("Evaluation Epoch {} has completed. Time: {}".format(
            epoch_id, epoch_time))
        return True

    def run_train_epoch(self, epoch_id: int,speck):
        self.model.train()

        start_time = time.time()
        mean_loss = 0
        step_count = 0
        for step_id,ref in enumerate(self.train_data_loader):
            loss,loglike_loss,bias_loss = self.train_step(ref,speck)
            epoch_time = time.time() - start_time

            mean_loss += loss
            step_count += 1

            self.log_train_step(epoch_id, step_id, epoch_time,loss,bias_loss,loglike_loss)

        epoch_time = time.time() - start_time
        mean_loss /= max(step_count, 1)

        self.log_train_epoch(epoch_id, epoch_time)

        return epoch_time, mean_loss

    def run_evaluation_epoch(self, epoch_id: int,denorm):
        self.model.eval()

        step_count = 0
        start_time = time.time()

        with torch.no_grad():
            for step_id, (input_data,name) in enumerate(self.eval_data_loader):
                model_output = self.predict_step(input_data,name,epoch_id,step_id,denorm)
                
                step_count += 1
                epoch_time = time.time() - start_time

                self.log_evaluation_step(epoch_id, step_id, epoch_time)

        epoch_time = time.time() - start_time

        self.log_evaluation_epoch(epoch_id, epoch_time)

        return epoch_time

    def load_optimizer_state(self):
        if not self.state_storage.has_key("learning_rates"):
            return

        learning_rates = self.state_storage.get_value("learning_rates")

        for learning_rate, param_group in zip(learning_rates, self.optimizer.param_groups):
            param_group["lr"] = learning_rate

    def save_optimizer_state(self):
        learning_rates = []
        for param_group in self.optimizer.param_groups:
            learning_rates.append(float(param_group['lr']))

        self.state_storage.set_value("learning_rates", learning_rates)

    def save_last_model(self, phase, epoch_id):
        os.makedirs(self.model_save_path, exist_ok=True)
        model_path = os.path.join(self.model_save_path, "{}_epoch_{}".format(phase,epoch_id))
        save_model(self.model, model_path)
        LOGGER.info("Model was saved in {}".format(model_path))

    def load_last_model(self, phase, epoch_id):
        last_model_path = os.path.join(self.model_save_path, "{}_epoch_{}".format(phase,epoch_id))
        load_model(self.model, last_model_path)

    def run(self):
        denorm = denormalization(self.norm[0],self.norm[1],self.norm[2])
        speck = add_speckle_pytorch(self.norm[0],self.norm[1])
        
        os.makedirs('./data/sample', exist_ok=True)
        
        start_epoch_id = 0

        if self.state_storage.has_key("start_epoch_id") and self.state_storage.has_key("phase"):
            last_saved_epoch = self.state_storage.get_value("start_epoch_id")-1
            last_saved_phase = self.state_storage.get_value("phase")
            if last_saved_phase == self.phase:
                start_epoch_id = last_saved_epoch+1
            try:
                self.load_last_model(last_saved_phase, last_saved_epoch)
                LOGGER.info("Last saved weights : phase {}, epoch {}".format(last_saved_phase,last_saved_epoch))
                LOGGER.info("Starting training with parameters : phase {}, epoch {}".format(self.phase,start_epoch_id))
                            
            except:
                LOGGER.exception("Exception occurs during loading a model. Starting to train a model from scratch...")
                
        else:
            LOGGER.info("Model not found in {}. Starting to train a model from scratch...".format(self.model_save_path))
        
        print(self.phase)
        self.state_storage.set_value("phase", self.phase)
        self.load_optimizer_state()

        epoch_id = start_epoch_id
        while self.epoch_count is None or epoch_id < self.epoch_count:
            
            _, mean_train_loss = self.run_train_epoch(epoch_id,speck)
            self.run_evaluation_epoch(epoch_id,denorm)
            self.scheduler.step()

            self.state_storage.set_value("start_epoch_id", epoch_id + 1)
            self.save_optimizer_state()
            self.save_last_model(self.phase,epoch_id)

            epoch_id += 1
