# %% import
import os
import numpy as np
import pandas as pd

def clamp(x, a, b):
  if type(x) is np.ndarray or type(b) is np.ndarray:
    return np.clip(x, a, b)
  return max(a, min(x, b))

read_excel = pd.read_excel