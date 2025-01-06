import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Union, Dict
import copy
import tqdm
try:
    from .constants import *
    from .containers import Data, Dataset
    from .model_pipe import ModelPipe, Path
    from .loads import save_file, load_file
    from .post_process import post_process, portfolio_post_process

except ImportError:
    from constants import *
    from containers import Data, Dataset
    from model_pipe import ModelPipe, Path
    from loads import save_file, load_file
    from post_process import post_process, portfolio_post_process
    



class Workflow:
    def __init__(self, dataset:Dataset, model_pipe:ModelPipe):
        self.dataset = dataset
        self.model_pipe = model_pipe
        # check keys
        assert self.dataset.keys() == self.model_pipe.keys(),"dataset and model_pipe must have the same keys"
        self.cvbt_paths = None
        self.test_path = None

    def set_dataset(self, dataset:Dataset):
        self.dataset = dataset
        return self

    def set_model_pipe(self, model_pipe: ModelPipe):
        self.model_pipe = model_pipe
        return self

    def cvbt(self, k_folds:int = 4, seq_path:bool = False, start_fold:int = 0, n_paths:int = 4, burn_fraction:float = 0.1, min_burn_points:int = 3):        
        # create folds splits
        self.dataset.split_ts(k_folds)                  
        start_fold = max(1, start_fold) if seq_path else start_fold     
        self.cvbt_paths = []
        for m in tqdm.tqdm(range(n_paths)):
            cvbt_path = Path()
            for fold_index in range(start_fold, k_folds):  
                # build train test split
                train_test = self.dataset.train_test_split(
                                                    test_fold_idx = fold_index, 
                                                    burn_fraction = burn_fraction, 
                                                    min_burn_points = min_burn_points, 
                                                    seq_path = seq_path
                                                    )
                # copy model pipe
                local_model_pipe = copy.deepcopy(self.model_pipe)
                # assign data to model pipe
                for e in train_test: 
                    # if there is no data, remove the model from the pipe
                    if e.get('train_data').empty or e.get('test_data').empty:
                        local_model_pipe.remove(e.get('key'))
                    else:
                        local_model_pipe.set_data(**e)  
                # estimate model
                local_model_pipe.estimate()
                # evaluate model
                local_model_pipe.evaluate()
                # add to path
                cvbt_path.add(local_model_pipe)
            cvbt_path.join()
            self.cvbt_paths.append(cvbt_path.get_results())
        self.paths = self.cvbt_paths
        return self.cvbt_paths

    # call functions from post process that are implemented on other file
    def post_process(self, pct_fee = 0., seq_fees = False, sr_mult = 1, n_boot = 1000, key = None, start_date = '', end_date = '', output_paths:bool = True):
        return post_process(
                    paths = self.paths, 
                    pct_fee = pct_fee,
                    seq_fees = seq_fees,
                    sr_mult = sr_mult,
                    n_boot = n_boot,
                    key = key,
                    start_date = start_date,
                    end_date = end_date,
                    output_paths = output_paths
                    )

    def portfolio_post_process(self, pct_fee = 0., seq_fees = False, sr_mult = 1,n_boot = 1000, view_weights = True, use_pw = True, multiplier = 1, start_date = '', end_date = '', n_boot_datasets:int = None, output_paths:bool = True):
        return portfolio_post_process(
                                paths = self.paths, 
                                pct_fee = pct_fee,
                                seq_fees = seq_fees,
                                sr_mult = sr_mult,
                                n_boot = n_boot,
                                view_weights = view_weights,
                                use_pw = use_pw,
                                multiplier = multiplier,
                                start_date = start_date,
                                end_date = end_date,
                                n_boot_datasets = n_boot_datasets,
                                output_paths = output_paths
                                )

    # main methods to make studies on the dataset
    def train(self):
        # associate model pipes with the dataset
        for key, data in self.dataset.items():
            self.model_pipe.set_train_data(key, data) 
        # estimate models
        self.model_pipe.estimate()
        return self.model_pipe

    def test(self):    
        self.test_path = Path()
        for key, data in self.dataset.items():
            self.model_pipe.set_test_data(key, data)
        self.model_pipe.evaluate()
        self.test_path.add(self.model_pipe)
        self.test_path.join()
        self.test_path = [self.test_path.get_results()]
        self.paths = self.test_path
        return self.test_path


    def live(self):
        out = {}
        for key, data in self.dataset.items():
            if self.model_pipe.has_keys(key):
                # note
                # for a live application it makes sense that the last observation of 
                # y has nan because it is not available yet!
                # assume here that the data has that format             
                xq_ = None
                x_ = None
                z_ = None
                if data.has_x:
                    xq_ = data.x[-1]
                    x_ = data.x[:-1]
                if data.has_z:
                    z_ = data.z[-1]
                w = self.model_pipe[key].get_weight(
                                        xq = xq_, 
                                        x = x_, 
                                        y = data.y[:-1], 
                                        z = z_,
                                        apply_transform_x = True, 
                                        apply_transform_y = True
                                        )
                pw = self.model_pipe[key].get_pw(data.y[:-1])       
                out.update({key:{'w':w,'pw':pw}})       
        # compute pw sum
        pw_sum = 0
        for k,v in self.model_pipe.items():             
            pw_sum+=v.train_pw
        out.update({'total_pw':pw_sum})
        return out





