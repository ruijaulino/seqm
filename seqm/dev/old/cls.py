import numpy as np
from abc import ABC, abstractmethod

class Weight:
    def __init__(self, w:np.ndarray):
        self.w = w

# Model class template
class BaseModel(ABC):
    
    @abstractmethod
    def estimate(self,y: np.ndarray, **kwargs):
        """Subclasses must implement this method"""
        pass

    @abstractmethod
    def get_weight(self, **kwargs) -> Weight:
        """Subclasses must implement this method"""
        pass

# Portfolio Model class template
class BasePortfolioModel(ABC):

    @abstractmethod
    def view(self):
        pass

    @abstractmethod
    def estimate(self, **kwargs):
        """Subclasses must implement this method"""
        pass

# Data Transform template
class BaseTransform(ABC):
    
    @abstractmethod
    def view(self):
        pass

    @abstractmethod
    def estimate(self, **kwargs):
        """Subclasses must implement this method"""
        pass

    @abstractmethod
    def transform(self, **kwargs):
        """Subclasses must implement this method"""
        pass
    
    @abstractmethod
    def inverse_transform(self, **kwargs):
        """Subclasses must implement this method"""
        pass





if __name__ == '__main__':
    model_pipe = ModelPipe()

    model_pipe.estimate()

