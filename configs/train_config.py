from pipeline.storage.state import StateStorageFile
from pipeline.scheduler.base import SchedulerWrapperIdentity
from pipeline.datasets.base import EmptyDataset
import torch
import os


class ConfigUnsupervised:
    def __init__(
            self,
            model,
            model_save_path,
            train_dataset,
            optimizer,
            loss,
            trainer,
            device=None,
            eval_dataset=None,
            scheduler=None,
            batch_size=1,
            num_workers=0,
            epoch_count=None,
            print_frequency=1,
            state_storage=None):

        if eval_dataset is None:
            eval_dataset = EmptyDataset()

        if scheduler is None:
            scheduler = SchedulerWrapperIdentity()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if state_storage is None:
            state_storage = StateStorageFile(os.path.join(model_save_path, "state"))

        self.model = model
        self.model_save_path = model_save_path
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.scheduler = scheduler
        self.loss = loss
        self.optimizer = optimizer
        self.epoch_count = epoch_count
        self.print_frequency = print_frequency
        self.trainer = trainer
        self.device = device
        self.state_storage = state_storage



