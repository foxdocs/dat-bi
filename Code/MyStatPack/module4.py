# Visualisation

import seaborn as sbs
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np

def plot_hist(data):
    data.hist()
    return

def plot_scatter(data):
    data.plot.scatter(x='col_0', y='col_1',  colormap='viridis')
    return

def plot_box(data):
    # compare in box-plot
    ax = data.boxplot(column='col_1', figsize=(4, 4))
    ax.set_xlabel('x')
    ax.set_ylabel('y')    
    return ax

def plot_bar(data):
    plot = sbs.barplot(x='col_0', y='col_1', data = data)
    return plot

def plot_line(data):
    plt.plot(sorted(data['col_0']), sorted(data['col_1']), marker = '*')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Demo Plot')
    plt.legend('x', 'y')

# see more
# https://www.simplilearn.com/tutorials/python-tutorial/data-visualization-in-python