from pipeline.utils import run_train,load_config
import sys

def main():
    
    train_dir = './data/train_A'
    eval_dir = './data/eval'
    norm = [-2.0,9.5,0]
    alphas = [420]
    config = load_config('configs/SAR2SAR_A_supervised','ConfigSAR2SAR_A_supervised')
    

    # Try different weight for the regularizatio parameter
    for alpha in alphas:
        conf = config(train_dir,eval_dir,norm,alpha_ = alpha,save_dir = './pipeline/out/supervised_{}'.format(alpha))
        run_train(conf)
      
    # # Only one weight  
    # alpha = 1
    # conf = config(train_dir,eval_dir,norm,alpha_ = alpha,save_dir = './pipeline/out/supervised_only')
    # run_train(conf)

if __name__ == "__main__":
    main()