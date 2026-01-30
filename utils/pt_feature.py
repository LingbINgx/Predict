import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np


class pt_feature():
    def __init__(self, data, axes_shape:tuple[int, int]=(3, 2), figsize:tuple[int, int]=(15, 8)):
        self.data = data
        self.i = 0
        self.j = 0
        self.fig, self.axes = plt.subplots(*axes_shape, figsize=figsize) 

    def plot_feature(self, feature_name):
        i = self.i
        j = self.j
        ax = self.axes[i, j]
        
        x = self.data[feature_name].to_numpy()
        ax.hist(x, bins=30, alpha=0.7, density=True, color='blue', edgecolor='black', zorder=1, label=feature_name)
        
        mu, std = norm.fit(x)
        x_fit = np.linspace(x.min(), x.max(), 1000)
        p = norm.pdf(x_fit, mu, std)
        ax.plot(x_fit, p, linewidth=2, zorder=2, color='red', label=f'Fit: $\\mu$={mu:.2f}, $\\sigma$={std:.2f}')
        
        ax.grid(True, alpha=0.6, linestyle='--', linewidth=0.6)
        ax.set_title(feature_name)
        ax.legend()
        self.fig.tight_layout()
        if self.j == self.axes.shape[1] - 1:
            self.j = 0
            self.i += 1
        else:
            self.j += 1
        