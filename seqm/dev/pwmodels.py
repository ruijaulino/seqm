import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

class BasePWModel(ABC):
    
    @abstractmethod
    def view(self) -> None:
        pass

    @abstractmethod
    def fit(self,arr: np.ndarray) -> 'BasePWModel':
        """Subclasses must implement this method"""
        pass

    @abstractmethod
    def pw(self, arr: np.ndarray) -> float:
        """Subclasses must implement this method"""
        pass

class InvVolPw(BasePWModel):
    def __init__(self):
        self.std = None   

    def view(self):
        print('** InvVolPw **')
        print('Scale: ', self.std)
    
    def pw(self, y, s):
        return np.sqrt(np.sum(np.power(1/self.std,2)))
    
    def fit(self, y, s):
        """Compute the mean and standard deviation of the data."""
        self.std = np.std(y, axis=0)

class SRPW(BasePWModel):
    def __init__(self):
        self.sr = None   

    def view(self):
        print('** SRPW **')
        print('SR: ', self.sr)
    
    def pw(self, y, s):
        return self.sr
    
    def fit(self, y, s):
        """Compute the mean and standard deviation of the data."""
        self.sr = np.mean(s) / np.std(s)



