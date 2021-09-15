import torch.utils.data as data
import torch
import numpy as np
import glob
import random
import os


class test_data(data.Dataset):
    """ Store the eval images (1xHxWx1) and get normalized version"""
    def __init__(self, dataset, process_func=None):
        self.dataset_ = dataset
        self.Transform = process_func

    def __len__(self):
        return len(self.dataset_)

    def __getitem__(self, item):
        path = self.dataset_[item]
        name = os.path.basename(path)
        name = os.path.splitext(name)[0]
            
        im = np.load(path)
        im = im[:,:,np.newaxis]
        x = self.Transform(im)
        return x,name
    
class train_A_sup(data.Dataset):
    """ Store the patch (NbxHxWx1) and get 2 different normalized noisy version for training"""
    def __init__(self, dataset, process_func=None,process_ref=None):
        self.dataset_ = dataset
        self.Transform = process_func
        self.Transform_ref = process_ref

    def __len__(self):
        return len(self.dataset_)

    def __getitem__(self, item):
        x = self.Transform(self.dataset_[item])
        y = self.Transform(self.dataset_[item])
        ref = self.Transform_ref(self.dataset_[item])
        return (x,y,ref)
    
class train_A_sup_bias(data.Dataset):
    """ Store the patch (NbxHxWx1) and get 2 different normalized noisy version for training"""
    def __init__(self, dataset, process_func=None):
        self.dataset_ = dataset
        self.Transform = process_func

    def __len__(self):
        return len(self.dataset_)

    def __getitem__(self, item):
        ref = self.Transform(self.dataset_[item])
        return ref
    
class train_A(data.Dataset):
    """ Store the patch (NbxHxWx1) and get 2 different normalized noisy version for training"""
    def __init__(self, dataset, process_func=None):
        self.dataset_ = dataset
        self.Transform = process_func

    def __len__(self):
        return len(self.dataset_)

    def __getitem__(self, item):
        x = self.Transform(self.dataset_[item])
        y = self.Transform(self.dataset_[item])
        return (x,y)

class train_BC(data.Dataset):
    """ Store the patch (nb_piles,nb_patches,H,W,1,nb_images/pile) and get noisy and other noisy with change correction"""
    def __init__(self, noisy, denoised, process_func=None):
        
        # Get number of patches and random indexes for reference and input image in each pile 
        len_ = []
        index_ = []
        for i in range(len(noisy)):
            
            # Number of patches in pile i
            nb_patches = noisy[i].shape[0]
            len_.append(nb_patches)
            
            # Get a list of random int (i,j) with i the image index and j the distance beetween the two images
            nb_im = noisy[i].shape[4]
            list_s = np.random.randint(0,nb_im,size=(1,nb_patches)) # index of image one
            list_d = np.random.randint(1,nb_im,size=(1,nb_patches)) # distance beetween the images
            list_d = (list_s+list_d)%nb_im # index of the 2nd images
            indexes = np.concatenate((list_s,list_d),axis=0)
            index_.append(indexes)
        
        self.index_ = index_
        self.len_ = len_
        self.noisy_ = noisy
        self.denoised_ = denoised
        self.Transform = process_func
        
    def __len__(self):
        return sum(self.len_)

    def __getitem__(self, item):
        
        # get the pile and the index in pile
        index_patch = item
        patch_in_image = self.len_[0]
        pile = 0
        while index_patch > patch_in_image-1:
            index_patch -= patch_in_image
            pile += 1
            patch_in_image = self.len_[pile]
            
        index_input = self.index_[pile][0,index_patch]
        index_ref = self.index_[pile][1,index_patch]
        
        inp = self.noisy_[pile][index_patch,:,:,:,index_input]
        ref = self.noisy_[pile][index_patch,:,:,:,index_ref]
        denoised_inp = self.denoised_[pile][index_patch,:,:,:,index_input]
        denoised_ref = self.denoised_[pile][index_patch,:,:,:,index_ref]
        change_comp = np.multiply(np.divide(ref,denoised_ref+10**(-3)),denoised_inp)
        
#         change_comp = np.clip(ref-denoised_ref+denoised_inp,a_min=0,a_max=None)
#         change_comp = ref-denoised_ref+denoised_inp
        
#         print('item : {}, index patch : {}, pile : {}'.format(item,index_patch,pile))
#         np.save('./data/debug/inp.npy',inp)
#         np.save('./data/debug/ref.npy',ref)
#         np.save('./data/debug/denoised_inp.npy',denoised_inp)
#         np.save('./data/debug/denoised_ref.npy',denoised_ref)
#         np.save('./data/debug/change_comp.npy',change_comp)
        
        return (self.Transform(inp),self.Transform(change_comp))


    
    
    
    
    
    
    
    
    