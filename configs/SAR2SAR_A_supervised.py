from configs.train_config import ConfigUnsupervised
import os

from pipeline.datasets.load import generate_patches_A, load_eval_data
from pipeline.datasets.datasets import train_A_sup,test_data
from pipeline.datasets.preprocessing import *

from torch.utils.data import DataLoader
from torch.nn import DataParallel
from torchvision import transforms
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

from pipeline.models.unet import UNet

from pipeline.loss.perceptual_loss import SAR2SARPerceptualLoss,SAR2SARPerceptualLoss_only

from pipeline.trainer.trainer_supervised import TrainerSupervised

from pipeline.storage.state import StateStorageFile

class ConfigSAR2SAR_A_supervised(ConfigUnsupervised):
    def __init__(self,train_dir,eval_dir,norm,alpha_=1,batch_size=4,save_dir='./pipeline/out'):
        
        patches = generate_patches_A(train_dir)
        process_A = transforms.Compose([add_speckle(),normalization(norm[0],norm[1],norm[2]),ToTensor()])
        process_ref = transforms.Compose([normalization(norm[0],norm[1],norm[2]),ToTensor()])
        train_patches = train_A_sup(patches,process_A,process_ref)
        
        eval = load_eval_data(eval_dir)
        process_eval = transforms.Compose([normalization(norm[0],norm[1],norm[2]),ToTensor()])
        eval_dataset = test_data(eval,process_eval)
        
        
        model = UNet()
        INIT_LR = 1e-3
        epochs = 50
        print_frequency = 100
        opt = Adam(model.parameters(), lr=INIT_LR,weight_decay=0.1)
        scheduler = MultiStepLR(opt,milestones=[20,40],gamma=0.1)
        device = 'cuda'
        save_weights = save_dir
        state_storage = StateStorageFile(os.path.join(save_weights, "state"))
        num_workers = 0
        loss = SAR2SARPerceptualLoss(norm[0],norm[1],device,alpha=alpha_)
        trainer = TrainerSupervised
        
        self.norm = norm
        self.phase = 'A'
                
        super().__init__(model=model,
                         model_save_path=save_weights,
                         train_dataset=train_patches,
                         optimizer=opt,
                         loss=loss,
                         trainer=trainer,
                         device=device,
                         eval_dataset=eval_dataset,
                         scheduler=scheduler,
                         batch_size=batch_size,
                         num_workers=num_workers,
                         epoch_count=epochs,
                         print_frequency=print_frequency,
                         state_storage=state_storage)