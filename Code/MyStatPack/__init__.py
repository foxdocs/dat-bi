import numpy as np
import pandas as pd

__all__ = ['module1', 'module2', 'module3', 'module4']

from . import module1, module2, module3, module4


def data(start, stop, rows, cols):
    # np.random.uniform(start,stop,(rows,columns)
    df = pd.DataFrame(np.random.randint(start,stop,(rows,cols)))    
    df = df.add_prefix('col')
    return df

def bell(median, stdiv, rows, cols):
    df = pd.DataFrame(np.random.normal(median, stdiv, (rows,cols)))
    df = df.add_prefix('col')
    return df