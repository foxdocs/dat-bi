# Correlation

import seaborn as sbs
import matplotlib.pyplot as plt

# correlation matrix
def my_corr(data):
    cormat = data.corr()
    return cormat


def my_corr_plot(cormat):
    sbs.heatmap(cormat, cmap = 'viridis',  annot=True, fmt=".2f", square=True, linewidths=.2)
    plt.show()