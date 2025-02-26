import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import List, Union, Dict
import copy
import tqdm



# Transform class
class Transform(ABC):
    
    @abstractmethod
    def view(self):
        pass

    @abstractmethod
    def estimate(self, arr:np.ndarray, **kwargs):
        """Subclasses must implement this method"""
        pass

    @abstractmethod
    def transform(self, arr:np.ndarray, **kwargs):
        """Subclasses must implement this method"""
        pass
    
    @abstractmethod
    def inverse_transform(self, arr:np.ndarray, **kwargs):
        """Subclasses must implement this method"""
        pass


# Transforms class: a dict of transforms
class Transforms(dict):

    def add(self, var:str, transform:Transform):
        self[var] = transform

    def estimate(self, data:'Data'):
        for k, v in self.items():
            v.estimate(getattr(data, k))


if __name__ == '__main__':
    pass