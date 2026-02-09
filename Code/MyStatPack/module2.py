# Dispersion
import math
import numpy as np
from typing import List
from . import module1

# range
def my_range(data: List) -> float:
    return max(data) - min(data)

# quantiles
def my_quantiles(data: List, proc: int) -> float:
    n = len(data)
    p = int(proc/100*n)
    return sorted(data)[p]


# variance
def my_variance(data: List) -> float:
    data_mean = module1.my_mean(data)
    n = len(data)

    # calculate everyone's distance from mean (deviation)
    data_dev = [np.square(x - data_mean) for x in data]
    data_dev_sum = sum(data_dev)
    return data_dev_sum / (n-1)


# standard deviation
def my_std(data: List) -> float:
    return math.sqrt(my_variance(data))

# outliers by interquartile range (IQR)
def my_outliers(data: List) -> float:
   q1 = my_quantiles(data, 25)
   q3 = my_quantiles(data, 75)
   IQR = q3 - q1
   # values smaller than 1.5 IQR below q1 and bigger that 1.5 IQR over q3 
   outliers = data[((data<(q1-1.5*IQR)) | (data>(q3+1.5*IQR)))]
   return outliers