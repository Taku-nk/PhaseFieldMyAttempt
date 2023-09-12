""" Plot loss history of DEM trianing.

Typical usage:
    from utils.visualize_loss_history import plot_loss
    save_dir = Path('C:/Users/Me/Desktop/')
    plot_loss(save_dir/"loss.csv", save_path=save_dir/"loss.svg")
"""

from pathlib import Path
import numpy as np
import pandas as pd

import sys
sys.path.append("C:\\Users\\taku\\Hiroshima-U-Master\\OneDrive - Hiroshima University\\ドキュメント\\1kouza\\MasterResearch\\PythonCode\\plot_2d")
from plot_2d import FigPlot, clip_df




def plot_loss(file_path, save_path=''):
    """ Plot DEM analysis loss.

    Args: 
        file_path: path like. File must be a .csv file with columns of 
            i, loss, id. id=0 means Adam, id=1 means L-BFGS-B
        save_path: path like. the save path.
            

    Returns: 
        None
    """
    df = pd.read_csv(file_path)
    i_adam = df.loc[df['id']==0, 'i'].to_numpy()
    loss_adam = df.loc[df['id']==0, 'loss'].to_numpy()

    i_lbfbs = df.loc[df['id']==1, 'i'].to_numpy()
    loss_lbfgs = df.loc[df['id']==1, 'loss'].to_numpy()




    fig_plot = FigPlot(figsize=(4,3), fontsize=12)

    offset_min = 0.05 * abs(loss_adam.max() - loss_lbfgs.min())
    fig_plot.ax.set_ylim(loss_lbfgs.min() - offset_min , loss_adam.max())

    fig_plot.ax.set_xlabel(r'Iteration')
    fig_plot.ax.set_ylabel(r'Loss')

    fig_plot.ax.locator_params(axis='x',nbins=4)
    fig_plot.ax.locator_params(axis='y',nbins=4)


    fig_plot.ax.plot(i_adam, loss_adam, label='Loss (Adam)', c='tab:blue', ls='-')
    fig_plot.ax.plot(i_lbfbs, loss_lbfgs, label='Loss (L-BFGS-B)', c='tab:orange', ls='-')

    if save_path != '':
        fig_plot.fig.savefig(save_path, bbox_inches='tight')

    fig_plot.show()


if __name__ == '__main__':
    plot_loss("./test_single_notch/output/iter_1/loss.csv")
    # plot_loss("./test_single_notch/output/iter_2/loss.csv")

