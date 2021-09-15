from pipeline.utils import run_predict,load_config
import sys
import glob
import os

""" input : folder with .npy images in amplitude
    output : folder with denoised images """


""" Prediction with the network version stored in weights_dir """
def main(eval_dir = './data/real',  # dataset for evaluation
         save_dir = './data/results/real', # where the result is stored
         weights_dir = './pipeline/out/unsupervised', # weights of thhe networks
        print_freq = 1): # Print in the logger for each image
    
    norm = [-2.0,9.5,0] 
    config = load_config('configs/SAR2SAR_pred','ConfigSAR2SAR_pred')
    
    conf = config(eval_dir,save_dir,weights_dir,norm,add_speck = False,print_freq = print_freq ) # Put add_speck = True for adding speckle on image 
    run_predict(conf)

""" Predictions for each version of the network in pipeline/out (with a different dataset for versions step A and step B,C""" 
def main_loop():
    
    
    weights_dirs = './pipeline/out'
    norm = [-2.0,9.5,0]
    config = load_config('configs/SAR2SAR_pred','ConfigSAR2SAR_pred')
    
    
    dirpaths = glob.glob(weights_dirs+'/*') # get all the model versions
    
    for dir_ in dirpaths: 
        dir_name = os.path.basename(dir_)
        save_dir = os.path.join('./data/results',dir_name)
        if '/sup' in dir_ or '_A' in dir_: # use real data or artificial data
            eval_dir = './data/eval'
            dtyp = 'artificial'
        else:
            eval_dir = './data/eval_real'
            dtyp = 'real'
            
        print('\nPredictions with {} data and weights : {}'.format(dtyp,dir_name))
        conf = config(eval_dir,save_dir,dir_,norm,add_speck = False,print_freq = 1 )
        run_predict(conf)
        
    
    
    
if __name__ == "__main__":
    main()