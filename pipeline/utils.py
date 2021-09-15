import os
import importlib
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.nn import DataParallel

from .logger import setup_logger


# Train a network with his config
def run_train(config):
    
    train_data = DataLoader(config.train_dataset,
                            batch_size=config.batch_size,
                            shuffle=True,
                            num_workers=config.num_workers)
    
    eval_data = DataLoader(config.eval_dataset)
    
    model_save_path = config.model_save_path
    os.makedirs(model_save_path, exist_ok=True)
    
    logger_path = os.path.join(model_save_path, "log.txt")
    setup_logger(out_file=logger_path)
    
    trainer = config.trainer(model=config.model,
                                  train_data_loader=train_data,
                                  eval_data_loader=eval_data,
                                  epoch_count=config.epoch_count,
                                  optimizer=config.optimizer,
                                  scheduler=config.scheduler,
                                  loss=config.loss,
                                  print_frequency=config.print_frequency,
                                  device=config.device,
                                  model_save_path=model_save_path,
                                  state_storage=config.state_storage,
                                  norm = config.norm,
                                  phase = config.phase)
    trainer.run()
    
# Make predictions with a network in a config
def run_predict(config):
    
    eval_data = DataLoader(
        config.dataset,
        batch_size=config.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=config.num_workers)

    model_save_path = config.model_save_path
    assert os.path.exists(model_save_path), "{} does not exist".format(model_save_path)
    
    os.makedirs(config.save_im_path, exist_ok=True)

    logger_path = os.path.join(model_save_path, "log_predict.txt")
    setup_logger(out_file=logger_path)

    
    predictor = config.predictor(
        model=config.model,
        data_loader=eval_data,
        print_frequency=config.print_frequency,
        device=config.device,
        model_save_path=model_save_path,
        speckle = config.speckle,
        norm = config.norm,
        save_im_path = config.save_im_path)

    predictor.run()

# Load a configuration
def load_config(module_path, cls_name):
    module_path_fixed = module_path
    if module_path_fixed.endswith(".py"):
        module_path_fixed = module_path_fixed[:-3]
    module_path_fixed = module_path_fixed.replace("/", ".")
    module = importlib.import_module(module_path_fixed)
    assert hasattr(module, cls_name), "{} file should contain {} class".format(module_path, cls_name)

    cls = getattr(module, cls_name)
    return cls

# Save a model
def save_model(model, path):
    if isinstance(model, DataParallel):
        model = model.module
        
    with open(path, "wb") as fout:
        torch.save(model.state_dict(), fout)
        
# Load a model
def load_model(model, path):
    with open(path, "rb") as fin:
        state_dict = torch.load(fin)
        
    model.load_state_dict(state_dict)

# From cpu to gpu or the opposite
def move_to_device(tensor: list or tuple or torch.Tensor, device: str):
    if isinstance(tensor, list):
        return [move_to_device(elem, device=device) for elem in tensor]
    if isinstance(tensor, tuple):
        return (move_to_device(elem, device=device) for elem in tensor)
    return tensor.to(device)

# Apply a treshold (maybe np.clip is better), a defined treshold or mean+3*var
def tresh_im(img,treshold=None,k=3):
    imabs = np.abs(img)
    #imabs = img
    if treshold == None:
        mean = np.mean(imabs)
        std = np.std(imabs)
        treshold = mean+k*std
    imabs[imabs>treshold] = treshold
#     print('Treshold : ' + str(treshold))
    return imabs

# Plot an image (no treshold is applied)
def plot_im(img,title = ''):
    plt.imshow(img, cmap = 'gray')
    plt.axis('off')
    plt.title(title)
    plt.show()

# Plot an image with a treshold, if tresh is None it's an automatic treshold mean+3*var
def disp_sar(im,tresh=None):
    if tresh == None:
        im_t = tresh_im(im)
    else:
        im_t = tresh_im(im,treshold=tresh)
    plot_im(im_t)

# Save an image
def save_im(im,fold):
    im = (im-np.amin(im))*255/np.amax(im)
    cv2.imwrite(fold, im)
#     print('saving sample : {}'.format(fold))
    

if __name__ == "__main__":  
    pass