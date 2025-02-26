import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import List, Union, Dict
import copy
import tqdm


try:
    from .models import Model, PortfolioModel
    from .transforms import Transform, Transforms
    from .containers import Data, Dataset    
except ImportError:
    from models import Model, PortfolioModel
    from transforms import Transform, Transforms
    from containers import Data, Dataset



#if __package__ is None:
#    # running file standalone
#    print('Running in standalone')
#else:
#    print("Running as package")
#    # running from package
#    from seqm.models import Model, PortfolioModel
#    from seqm.transforms import Transform, Transforms
#    from seqm.containers import Data, Dataset

# ----------------------------------------------
# MODEL PIPELINE CLASSES

# ModelPipeUnit
# Handles a Model and Transforms applied to a Data object
class ModelPipeUnit:
    def __init__(self, model:Model = None, transforms:Transforms = None):
        self.model = model
        self.transforms = transforms
    
    def estimate(self, data:Data):
        # TO DO: transforms estimate and apply 
        # This will require a copy to the data...
        
        # model estimation is done with data as dict
        # it should have the required fields
        self.model.estimate(**data.as_dict())
    
    def get_weight(self, data:Data):
        # for a live setting, if features are present, we need an aditional
        # point on y (can be zeros) and this should be done when building the
        # data to send into the workflow. Just here the remainder

        # apply transforms (need to copy data...)
        # call get_weight from model with correct inputs
        
        # build q variables
        d = data.as_dict()
        d_add = {}
        for k, v in d.items():
            d_add.update({f"{k}q":v[-1]})
            d[k] = v[:-1]
        d.update(d_add)
        # get weight
        return self.model.get_weight(**d)
        
    def evaluate(self, data:Data):
        """Evaluate the model using the test data and return performance metrics."""
        # this will change fields s, weight_* in data object inplace        
        # iterate on data and run live        
                
        for i in range(data.n):       
            # this will filter data at index i suitable to make decision
            # it also filter by multisequence index in the .at method
            w = self.get_weight(data.at(i)) 
            data.w[i] = w
            data.s[i] = np.dot(data.y.at(i), w)
        return data

# List of ModelPipeUnits
# Objective here is to train a stack of model and then average predictions
# Works on top of Data
class ModelPipeStack(list):
    
    def add(self, model:Model = None, transforms:Transforms = None):
        self.append(ModelPipeUnit(model, transforms))
    
    def estimate(self, data:Data):
        # pipes cannot be empty!
        for unit in self: unit.estimate(data)
    
    def evaluate(self, data:Data):
        assert len(self) == 1, "ModelPipeStack for many model not implemented yet..."
        # # should be something like this
        # res = []
        # for pipe in self:
        #     tmp = pipe.evaluate(dataset.copy())
        #     res.append(tmp)
        # copied datasets were changed in place
        # take mean of weights to compute performance
        # dataset.w = mean of w in res
        
        # for now just consider single model version
        # data is changed inplace..
        self[0].evaluate(data)
        return data

# Dict of ModelPipeStack
# Objective here is to handle for several data in a dataset where
# each one has a ModelPipeStack associated

class ModelPipeContainer(dict):    
    
    def add(self, key:str, model:Model, transforms:Transforms = None):
        if key not in self:
            self[key] = ModelPipeStack()
        self[key].add(model, transforms)
    
    def estimate(self, dataset:Dataset):
        for k, data in dataset.items():
            assert k in self, "dataset contains a key that is not defined in ModelPipe. Exit.."
            self[k].estimate(data)
    
    def evaluate(self, dataset:Dataset):
        for k, data in dataset.items():
            assert k in self, "dataset contains a key that is not defined in ModelPipe. Exit.."            
            self[k].evaluate(data)   
        return dataset

# Full Model Pipe for several datasets
# also includes a portfolio model that decides how allocation
# works along datasets    

class ModelPipe():
    def __init__(self, portfolio_model:PortfolioModel = None):
        self.portfolio_model = portfolio_model
        self.model_pipe_container = ModelPipeContainer()
    
    def add(self, key, model, transforms = None):
        self.model_pipe_container.add(key, model, transforms)
    
    def estimate(self, dataset):        
        # dataset_dict is a dict of dataset        
        if self.portfolio_model:
            self.portfolio_model.estimate(dataset, self.model_pipe_container)
        self.model_pipe_container.estimate(dataset)

    def evaluate(self, dataset):
        # dataset_dict is a dict of dataset
        self.model_pipe_container.evaluate(dataset)
        # correct predictions/weights with portfolio model
              
        
        return dataset
    
# changes dataset in place
def cvbt_path(
            dataset:Dataset, 
            model_pipe:Union[ModelPipeContainer,ModelPipe],
            k_folds:int = 4, 
            seq_path:bool = False, 
            start_fold:int = 0, 
            burn_fraction:float = 0.1, 
            min_burn_points:int = 3
            ):
    
    dataset.split_ts(k_folds)                  
    start_fold = max(1, start_fold) if seq_path else start_fold     

    # tmp_model_pipes_map = {}
    for fold_index in range(start_fold, k_folds):  
        # build train test split
        train_dataset, test_dataset = dataset.split(
                                                    fold_index, 
                                                    burn_fraction = burn_fraction, 
                                                    min_burn_points = min_burn_points, 
                                                    seq_path = seq_path
                                                    )
        
        # copy model pipe
        # this is an operation without much overhead
        tmp_model_pipe = copy.deepcopy(model_pipe) 
        # train model
        tmp_model_pipe.estimate(dataset = train_dataset)
        # estimate model - the results will be written in dataset because .between uses simple indexing
        tmp_model_pipe.evaluate(test_dataset) #

        # set performance on dataset (maybe not needed because it will be already overriten! CHECK THIS)
    return dataset    


def linear(n=1000,a=0,b=0.1,start_date='2000-01-01'):
    x=np.random.normal(0,0.01,n)
    y=a+b*x+np.random.normal(0,0.01,n)
    dates=pd.date_range(start_date,periods=n,freq='D')
    data=pd.DataFrame(np.hstack((y[:,None],x[:,None])),columns=['y1','x1'],index=dates)
    return data


def test_estimate_1():

    df = linear(n=1000,a=0,b=0.1,start_date='2000-01-01')
    dataset = Dataset()
    dataset.add('dataset', df)

    from models import LR
    model = LR()

    model_pipe = ModelPipe()
    model_pipe.add('dataset', model)
    model_pipe.estimate(dataset)


    # evaluate
    
    df1 = linear(n=1000,a=0,b=0.1,start_date='2000-01-01')
    dataset1 = Dataset()
    dataset1.add('dataset', df1)

    model_pipe.evaluate(dataset1)
    print(dataset1)

def test_cvbt():

    df = linear(n=1000,a=0,b=0.1,start_date='2000-01-01')
    dataset = Dataset()
    dataset.add('dataset', df)

    from models import LR
    model = LR()

    model_pipe = ModelPipe()
    model_pipe.add('dataset', model)

    dataset = cvbt_path(
                dataset = dataset, 
                model_pipe = model_pipe
                )
    print(dataset)




if __name__ == '__main__':
    # test_estimate_1()
    test_cvbt()
    # test_memory()
    
