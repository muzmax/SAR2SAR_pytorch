import torch

class PredictConfigBase:
    def __init__(
            self,
            model,
            model_save_path,
            dataset,
            predictor,
            save_im_path = './data/results',
            device=None,
            batch_size=1,
            num_workers=0,
            print_frequency=1):
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = model
        self.dataset = dataset
        self.model_save_path = model_save_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.print_frequency = print_frequency
        self.predictor = predictor
        self.device = device
        self.save_im_path = save_im_path
