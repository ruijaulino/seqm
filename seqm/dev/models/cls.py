import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import List, Union, Dict
import copy
import tqdm

# Model class
class Model(ABC):


    @abstractmethod
    def estimate(self, y, x, z, msidx, **kwargs):
        """Subclasses must implement this method"""
        pass

    @abstractmethod
    def get_weight(self, xq, zq, x, y, z, msidx, *args, **kwargs):
        """Subclasses must implement this method"""
        # xq, zq must be seen as the last element in x, z!
        # need to implement this on the model!

        pass

class PortfolioModel(ABC):
    
    @abstractmethod
    def estimate(self, dataset:'Dataset', model_pipe:Union['ModelPipeContainer','ModelPipe']):
        """Subclasses must implement this method"""
        cvbt_path()
        pass




if __name__ == '__main__':
    pass