from pipeline.utils import run_train,load_config
import sys

""" not working because of the log fonction of pytorch """

def main():
    
    train_dir = './data/train_A'
    eval_dir = './data/eval'
    norm = [-2.0,9.5,0]
    alphas = [1]
    config = load_config('configs/SAR2SAR_A_supervised_bias','ConfigSAR2SAR_A_supervised_bias')
    

    
    for alpha in alphas:
        conf = config(train_dir,eval_dir,norm,alpha_ = alpha,save_dir = './pipeline/out/supervised_bias_{}'.format(alpha))
        run_train(conf)
        
if __name__ == "__main__":
    main()