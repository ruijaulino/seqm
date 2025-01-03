import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Union, Dict
import copy
import tqdm

try:
    
    from .workflow import Workflow
    from .containers import Dataset
    from .models import ConditionalGaussian, GaussianHMM
    from .model_pipe import ModelPipe
    from .transform import RollPWScaleTransform
    from .post_process import post_process,portfolio_post_process

except ImportError:

    from workflow import Workflow
    from containers import Dataset
    from models import ConditionalGaussian, GaussianHMM
    from model_pipe import ModelPipe
    from transform import RollPWScaleTransform
    from post_process import post_process,portfolio_post_process

def linear(n = 1000, a = 0, b = 0.1, start_date = '2000-01-01'):
    x = np.random.normal(0, 0.01, n)
    y = a + b*x + np.random.normal(0, 0.01, n)
    dates = pd.date_range(start_date, periods = n, freq = 'D')
    data = pd.DataFrame(np.hstack((y[:,None],x[:,None])), columns = ['y1', 'x1'], index = dates)
    return data

if __name__=='__main__':
    data1 = linear(n = 500, a = 0, b = 0.1, start_date = '2000-01-01')
    data2 = linear(n = 1000, a = 0, b = 0.1, start_date = '2003-01-01')
    # data3 = linear(n = 6000, a = 0, b = 0.1, start_date = '2001-01-01')


    # create dataset
    dataset = Dataset({'dataset 1':data1,'dataset 2':data2})   

    # create model pipe
    model=ConditionalGaussian(n_gibbs=None,kelly_std=3,max_w=1)        
    model_pipe = ModelPipe(master_model = model)    
    for key in dataset.keys():
        model = None
        # model = ConditionalGaussian(n_gibbs=None,kelly_std=3,max_w=1)
        # did not set individual model on purpose
        # but can be done (need to test this feature!)
        model_pipe.add(model = model, x_transform = RollPWScaleTransform(window=10), y_transform = RollPWScaleTransform(window=10), key = key)

    # create workflow
    workflow = Workflow(dataset = dataset, model_pipe = model_pipe)
    paths = workflow.cvbt(k_folds = 10, seq_path = True)
    
    workflow.train()
    workflow.test()

    # print(workflow.live())

    s = portfolio_post_process(workflow.cvbt_paths)
    print(s)
    # s = portfolio_post_process(paths, pct_fee = {'dataset 1':0, 'dataset 2':0.5,'dataset 3':0})
    # print(s)