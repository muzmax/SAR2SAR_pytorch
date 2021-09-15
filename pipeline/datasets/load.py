import glob
import random
import os
import numpy as np
import gc
import shutil

def data_augmentation(image, mode):
        if mode == 0:
            # original
            return image
        elif mode == 1:
            # flip up and down
            return np.flipud(image)
        elif mode == 2:
            # rotate counterwise 90 degree
            return np.rot90(image)
        elif mode == 3:
            # rotate 90 degree and flip up and down
            image = np.rot90(image)
            return np.flipud(image)
        elif mode == 4:
            # rotate 180 degree
            return np.rot90(image, k=2)
        elif mode == 5:
            # rotate 180 degree and flip
            image = np.rot90(image, k=2)
            return np.flipud(image)
        elif mode == 6:
            # rotate 270 degree
            return np.rot90(image, k=3)
        elif mode == 7:
            # rotate 270 degree and flip
            image = np.rot90(image, k=3)
            return np.flipud(image)
        
def generate_patches_A(src_dir="./data/train_A",pat_size=256,step=0,stride=16,bat_size=4,data_aug_times=1,num_pic=None):
        count = 0
        filepaths = glob.glob(src_dir + '/*.npy')
        print("number of training data = %d" % len(filepaths))

        # calculate the number of patches
        for i in range(len(filepaths)):
            img = np.load(filepaths[i])

            im_h = np.size(img, 0)
            im_w = np.size(img, 1)
            for x in range(0+step, (im_h - pat_size), stride):
                for y in range(0+step, (im_w - pat_size), stride):
                    count += 1
        origin_patch_num = count * data_aug_times

        if origin_patch_num % bat_size != 0:
            numPatches = (origin_patch_num / bat_size + 1) * bat_size
        else:
            numPatches = origin_patch_num
        print("total patches = %d , batch size = %d, total batches = %d" % \
              (numPatches, bat_size, numPatches / bat_size))

        # data matrix 4-D
        numPatches=int(numPatches)
        inputs = np.zeros((numPatches, pat_size, pat_size, 1), dtype="float32") # 3 im + superim + coherence


        count = 0
        # generate patches
        for i in range(len(filepaths)): #scan through images
            img = np.load(filepaths[i])
            img_s = img
            img_s = np.reshape(np.array(img_s, dtype="float32"),
                                  (np.size(img_s, 0), np.size(img_s, 1), 1))  # extend one dimension

            # If data_aug_times = 8 then perform them all, otherwise pick one at random or do nothing
            for j in range(data_aug_times):
                im_h = np.size(img, 0)
                im_w = np.size(img, 1)
                if data_aug_times == 8:
                    for x in range(0 + step, im_h - pat_size, stride):
                        for y in range(0 + step, im_w - pat_size, stride):
                            inputs[count, :, :, :] = data_augmentation(img_s[x:x + pat_size, y:y + pat_size, :], \
                                  j)
                            count += 1
                else:
                    for x in range(0 + step, im_h - pat_size, stride):
                        for y in range(0 + step, im_w - pat_size, stride):
                            # to pick one at random, uncomment this line and comment the one below
                            """inputs[count, :, :, :] = self.data_augmentation(img_s[x:x + pat_size, y:y + pat_size, :], \
                                                                          random.randint(0, 7))"""


                            inputs[count, :, :, :] = data_augmentation(img_s[x:x + pat_size, y:y + pat_size, :],0)

                            count += 1


        # pad the batch
        if count < numPatches:
            to_pad = numPatches - count
            inputs[-to_pad:, :, :, :] = inputs[:to_pad, :, :, :]
        
        return inputs
    
    
def generate_patches_BC(phase,weights,src_dir="./data/train_BC",
                        pat_size=256,step=0,stride=16,bat_size=4,data_aug_times=1,num_pic=None):
    
    from predict import main 
    import cv2
    
    """ ============================================== """
    """ Get a directory with noisy and denoised images """
    """ ============================================== """
    
    dirpaths = glob.glob(src_dir+'/*/') # get all the piles of images
    print('\nDenoising real data of {} pile in order to make change compensation for phase {}...'.format(len(dirpaths),phase))
    
    for dir_ in dirpaths: 
        dir_name = os.path.dirname(dir_)
        dir_name = os.path.basename(dir_name)
        
        filepaths = glob.glob(dir_+'*.npy') # get all images of one pile
        filepaths.sort()
        
        noisy_path = './pipeline/temp/noisy/{}'.format(dir_name)
        denoised_path = './pipeline/temp/denoised/{}'.format(dir_name)
        
        os.makedirs(noisy_path)
        os.makedirs(denoised_path)
        
        if phase == 'B':
            size = np.load(filepaths[0]).shape # for conserving the size after subres
            noisy_sub_path = './pipeline/temp/noisy_sub/{}'.format(dir_name)
            os.makedirs(noisy_sub_path)
            
        for file in filepaths:
            im = np.load(file)
            im_name = os.path.basename(file)
            im_name = os.path.splitext(im_name)[0]
            if phase == 'B':
                im_sub = im[::2,::2]
                np.save(noisy_sub_path+'/{}.npy'.format(im_name),im_sub)
            np.save(noisy_path+'/{}.npy'.format(im_name),im)
            
        print('\nDenoising of pile {}...'.format(dir_name))  
        if phase == 'B':
            main(eval_dir = noisy_sub_path, 
                 save_dir = denoised_path,
                 weights_dir = weights,
                print_freq = len(filepaths)+1)
        else:
            main(eval_dir = noisy_path, 
                 save_dir = denoised_path,
                 weights_dir = weights,
                print_freq = len(filepaths)+1)

        filepaths = glob.glob(denoised_path+'/*')
        for file in filepaths:
            if ('denoised_' not in file) or ('.npy' not in file):
                os.remove(file)
        if phase == 'B':
            filepaths = glob.glob(denoised_path+'/*.npy')
            for file in filepaths:
                im = np.load(file)
                im = cv2.resize(im,(size[1],size[0]),interpolation = cv2.INTER_LINEAR)
                np.save(file,im)
            
    """ =========================== """
    """ Get noisy and denoised list """
    """ =========================== """
    
    noisy_list = []
    denoised_list = []
    
    dirpaths = glob.glob('./pipeline/temp/noisy/*/')
    for dir_ in dirpaths:
        dir_name = os.path.dirname(dir_)
        dir_name = os.path.basename(dir_name)
        
        filepaths = glob.glob(dir_+'*.npy')
        size = np.load(filepaths[0]).shape
        noisy_stack = np.zeros((size[0],size[1],len(filepaths)))
        denoised_stack = np.zeros((size[0],size[1],len(filepaths)))
        for i,file in enumerate(filepaths):
            im_name = os.path.basename(file)
            
            noisy_im = np.load(file)
            noisy_stack[:,:,i] = noisy_im
            denoised_im = np.load('./pipeline/temp/denoised/{}/denoised_{}'.format(dir_name,im_name))
            denoised_stack[:,:,i] = denoised_im
            
        noisy_list.append((dir_name,noisy_stack))
        denoised_list.append((dir_name,denoised_stack))
        
    del denoised_stack,noisy_stack,noisy_im,denoised_im,im
    
    
    """ =========== """
    """ Get patches """
    """ =========== """
    
    nb_im = len(noisy_list)
    print("\nnumber of training data %d" % nb_im)
    
    # calculate the number of patches and create indexes
    count_ = np.zeros(nb_im, dtype=np.uint16)
    for i in range(nb_im):
        img = noisy_list[i][1][:,:,0]
        im_h = np.size(img, 0)
        im_w = np.size(img, 1)
        for x in range(0 + step, (im_h - pat_size), stride):
            for y in range(0 + step, (im_w - pat_size), stride):
                count_[i] += 1
    origin_patch_num = np.sum(count_)
    
    numPatches = origin_patch_num
    
    print("total patches = %d , batch size = %d, total batches = %d" % \
          (numPatches, bat_size, int(numPatches / bat_size)))
    
    real_inputs = []
    count = np.zeros(nb_im, dtype=np.uint16)
    # generate patches
    print('creating train dataset')
    for i in range(nb_im): # first scan through the training images
        multi_img = noisy_list[i][1].shape[2]
        inputs = np.zeros((count_[i], pat_size, pat_size, 1, multi_img), dtype=np.float64)
        for m in range(multi_img): # scan over the time dimension and append them, then start back from where it ends
            count[i] = 0
            img = noisy_list[i][1][:,:,m]
            img_s = img
            img_s = np.reshape(np.array(img_s, dtype="float64"),
                                  (np.size(img_s, 0), np.size(img_s, 1), 1))  # extend one dimension
            im_h = np.size(img, 0)
            im_w = np.size(img, 1)
            for x in range(0 + step, im_h - pat_size, stride):
                for y in range(0 + step, im_w - pat_size, stride):
                    inputs[count[i], :, :, :, m] = img_s[x:x + pat_size, y:y + pat_size, :]
                    count[i] += 1
        real_inputs.append(inputs)

    print('creating priors')
    # generate clean patches for prior knowledge of changes
    denoised_prior_inputs = []
    for i in range(nb_im):  # first scan through the training images
        multi_img = denoised_list[i][1].shape[2]
        prior_inputs = np.zeros((count_[i], pat_size, pat_size, 1, multi_img), dtype=np.float32)
        for m in range(multi_img):  # scan over the time dimension and append them, then start back from where it ends
            count[i] = 0
            img = denoised_list[i][1][:, :, m].astype(np.float32)
            img_s = img
            img_s = np.reshape(np.array(img_s, dtype=np.float32),
                               (np.size(img_s, 0), np.size(img_s, 1), 1))  # extend one dimension
            im_h = np.size(img, 0)
            im_w = np.size(img, 1)
            for x in range(0 + step, im_h - pat_size, stride):
                for y in range(0 + step, im_w - pat_size, stride):
                    prior_inputs[count[i], :, :, :, m] = img_s[x:x + pat_size, y:y + pat_size, :]
                    count[i] += 1
        denoised_prior_inputs.append(prior_inputs)
    
    shutil.rmtree('./pipeline/temp')
    
    # (nb piles, nb_patches, H, W, 1, nb images/pile)
    return real_inputs,denoised_prior_inputs

        
          
def load_eval_data(dir="./data/eval"):
    
    filepaths = glob.glob(dir + '/*.npy')
    nb = len(filepaths)
    print("Number of eval data = %d" %nb)
        
    return filepaths


    