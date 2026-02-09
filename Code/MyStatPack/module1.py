# Measures of Central Tendency
from typing import List
from collections import Counter

# mean
def my_mean(data:List) -> float:
    return sum(data)/len(data)
    # return data.mean()

# median
def my_median_odd(data: List) -> float:
    mid = len(data) // 2
    return sorted(data)[mid]


def my_median_even(data: List) -> float:
    mid = len(data) // 2
    sorted_data = sorted(data)
    return (sorted_data[mid] + sorted_data[mid-1]) / 2


def my_median(data: List) -> float:
        even = len(data) % 2 == 0
        median = my_median_even(data) if even else my_median_odd(data)
        return median

# mode
def my_mode(data:List) -> float:
    cnt = Counter(data)
    print(cnt)
    max_cnt= max(cnt.values())
    max_key = [key for key, val in cnt.items() if val == max_cnt]
    return max_key



    