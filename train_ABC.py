from pipeline.utils import run_train,load_config
import sys

def main():
    
    norm = [-2.0,9.5,0]
    
    """ Train step A """
    eval_dir = './data/eval'
    train_dir_A = './data/train_A'
    config = load_config('configs/SAR2SAR_A','ConfigSAR2SAR_A')
    conf = config(train_dir_A,eval_dir,norm)
    run_train(conf)
    
    """ Train step B and C """
    eval_dir = './data/eval_real'
    train_dir_BC = './data/train_BC'
    config = load_config('configs/SAR2SAR_BC','ConfigSAR2SAR_BC')
    
    conf = config(train_dir_BC,eval_dir,'B',norm) # B
    run_train(conf)
    
    conf = config(train_dir_BC,eval_dir,'C',norm) # C
    run_train(conf)
    
    
    
    
if __name__ == "__main__":
    main()

