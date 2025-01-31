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

    def cvbt(self, k_folds:int = 4, seq_path:bool = False, start_fold:int = 0, n_paths:int = 4, burn_fraction:float = 0.1, min_burn_points:int = 3, dataset:Dataset = None, model_pipe:ModelPipe = None):
        cvbt_paths = self._cvbt(
                                dataset = self.dataset,
                                model_pipe = self.model_pipe,
                                k_folds = k_folds,
                                seq_path = seq_path,
                                start_fold = start_fold,
                                n_paths = n_paths,
                                burn_fraction = burn_fraction,
                                min_burn_points = min_burn_points,
                                )

        self.paths = [e.get_results() for e in cvbt_paths]
        return self.paths        

    def _cvbt(self, dataset:Dataset, model_pipe:ModelPipe, k_folds:int, seq_path:bool, start_fold:int, n_paths:int, burn_fraction:float, min_burn_points:int):        
        # create folds splits
        cvbt_paths = []
        start_fold = max(1, start_fold) if seq_path else start_fold     
        for m in tqdm.tqdm(range(n_paths)):
        # for m in range(n_paths):
            # copy dataset for this path
            path_dataset = copy.deepcopy(dataset)
            path_dataset.split_ts(k_folds)                  
            cvbt_path = Path()
            # tmp_model_pipes_map = {}
            for fold_index in range(start_fold, k_folds):  
                # build train test split
                path_train_dataset, path_test_dataset = path_dataset.split(
                                                    test_fold_idx = fold_index, 
                                                    burn_fraction = burn_fraction, 
                                                    min_burn_points = min_burn_points, 
                                                    seq_path = seq_path
                                                    )
                
                # copy model pipe
                local_model_pipe = copy.deepcopy(model_pipe)
                # train model
                local_model_pipe = self._estimate(
                                                dataset = path_train_dataset, 
                                                model_pipe = local_model_pipe, 
                                                )
                            
                # -----
                # # assign data to model pipe
                # for e in train_test: 
                #     # if there is no data, remove the model from the pipe
                #     if e.get('train_data').empty or e.get('test_data').empty:
                #         local_model_pipe.remove(e.get('key'))
                #     else:
                #         local_model_pipe.set_data(**e)  
                # -----

                # estimate model
                local_model_pipe = self._evaluate(path_test_dataset, local_model_pipe)
                # evaluate model
                # local_model_pipe.evaluate()
                # add to path
                # set oos_s in path_dataset
                path_dataset.set_oos_s(local_model_pipe, fold_index)
                cvbt_path.add(local_model_pipe)
                # tmp_model_pipes_map[fold_index] = local_model_pipe



            # join cvbt path
            cvbt_path.join()

            cvbt_paths.append(cvbt_path)            

        return cvbt_paths


    def estimate(self):
        # when called externally may need to run cvbt to setup pw ??

        self.model_pipe = self._estimate(self.dataset, self.model_pipe)

    # can be used internally or with the workflow directly
    def _estimate(self, dataset:Dataset, model_pipe:ModelPipe):        
        for key, data in dataset.items():
            model_pipe.set_train_data(key, data) 
        # estimate models
        model_pipe.estimate()        
        return model_pipe

    def evaluate(self):
        self.model_pipe = self._evaluate(self.dataset, self.model_pipe)
        self.test_path = Path()
        self.test_path.add(self.model_pipe)
        self.test_path.join()
        self.test_path = [self.test_path.get_results()]
        self.paths = self.test_path
        return self.test_path

    def _evaluate(self, dataset:Dataset, model_pipe:ModelPipe):    
        # check if this is an internal call
        # when the inputs are defined it is understood
        # that the call to train is being made from cvbt
        # or other internal method
        for key, data in dataset.items():
            model_pipe.set_test_data(key, data)
        model_pipe.evaluate()        
        return model_pipe

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

    def portfolio_post_process(self, pct_fee = 0., seq_fees = False, sr_mult = 1,n_boot = 1000, view_weights = True, use_pw = True, multiplier = 1, start_date = '', end_date = '', n_boot_datasets:int = None, output_paths:bool = True, sr_pw:bool = False):
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
                                output_paths = output_paths,
                                sr_pw = sr_pw
                                )






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
                t_ = None
                z_ = None
                if data.has_x:
                    xq_ = data.x[-1]
                    x_ = data.x[:-1]
                if data.has_t:
                    t_ = data.t
                if data.has_z:
                    z_ = data.z[-1]
                w = self.model_pipe[key].get_weight(
                                        xq = xq_, 
                                        x = x_, 
                                        y = data.y[:-1], 
                                        z = z_,
                                        t = t_,
                                        apply_transform_x = True, 
                                        apply_transform_t = True,
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

