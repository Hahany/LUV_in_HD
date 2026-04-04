import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import argparse
import scipy.stats as stats
import matplotlib.font_manager as fm
import pandas as pd
from hd2d_src.HopTrack.core import *
from matplotlib.font_manager import FontProperties

from matplotlib.font_manager import FontProperties

font_prop = FontProperties(family='Times New Roman')
font_path = '/home/xiaochu/.fonts/times.ttf'
fm.fontManager.addfont(font_path)


def plot_norm_distribution(ax, data, nbins, label, color, x1, x2, mcolor=None,  pset=None, lt=None, alpha_list=None,
                           markers=None, fit=True, density=True, vline=True, ms=6, lw=2):
    # data can be multi-dimension
    if mcolor is None:
        mcolor = color
    if markers is None:
        markers = ['s', '^', 'o', 'd', '*', 'p', '+', '<', '>']
    if alpha_list is None:
        alpha_list = [1, 1, 1, 1]
    if lt is None:
        lt = ['-', '-', '-', '-']
    n_out = []
    ave_list = []
    mu_list = []
    sigma_list = []
    h_list = []
    for i in range(len(data)):
        ave = np.mean(data[i])
        edges = np.linspace(x1[i], x2[i], nbins[i])
        n, _ = np.histogram(data[i], bins=edges, density=density)
        # n_pd = pd.Series(n)
        # n_coarse = n_pd.rolling(window=3, center=True).mean()
        n_coarse = n
        print(np.sum(n_coarse) * (edges[1] - edges[0]))
        if ax:
            if pset is None:
                ax.plot(0.5 * (edges[:-1] + edges[1:]), n_coarse, marker=markers[i],
                        linestyle='None', markeredgecolor=color[i], fillstyle='full', markerfacecolor=color[i],
                        label=label[i], markersize=ms)
            elif i in pset:
                ax.plot(0.5 * (edges[:-1] + edges[1:]), n_coarse, marker=markers[i],
                        linestyle='None', markeredgecolor=color[i], fillstyle='full', markerfacecolor=mcolor[i],
                        label=label[i], markersize=ms)
        # ax.text(0.01, 0.75-0.15*i, f'Mean {lable[i]}: {ave:.04f}', transform=ax.transAxes, fontsize=10)
        # ax.legend()#prop=font_prop)
        mu, std = norm.fit(data[i])
        # xrange = np.linspace( ave - 0.1, ave+0.1, 1000)
        xrange = np.linspace(x1[i], x2[i], 1000)
        p = (norm.pdf(xrange, mu, std))
        # p = p*len(data[i])*(edges[-1]-edges[0])/len(edges)
        if ax and fit:
            if pset is None:
                ax.plot(xrange, p, color=color[i], linewidth=lw, linestyle=lt[i],
                        alpha=alpha_list[i])  # ($\mu={mu:.04f}, \sigma={std:.04f}$)
                # ax.set_xlim(np.min(x1), np.max(x2))
                # ax.legend()
            elif i in pset:
                ax.plot(xrange, p, color=color[i], linewidth=lw, linestyle=lt[i],
                        alpha=alpha_list[i])  # ($\mu={mu:.04f}, \sigma={std:.04f}$)
                # ax.set_xlim(np.min(x1), np.max(x2))
                # ax.legend()
        n_out.append(n)
        ave_list.append(ave)
        mu_list.append(mu)
        s = np.std(data[i], ddof=1)
        sigma_list.append(s)
        height = np.max(n_coarse) + 0.02
        h_list.append(height)
    height = np.max(h_list)
    if vline:
        for i in range(len(h_list)):
            if not pset is None:
                if i in pset:
                    ax.vlines(x=np.mean(data[i]), ymin=0, ymax=height, color=color[i], ls='--')
                    #plot the standard error
                    y0 = 0.1
                    ax.vlines(x=ave_list[i] - (sigma_list[i] / np.sqrt(len(data[i]))), ymin=y0 - 0.005, ymax=y0 + 0.005,
                              color=color[i], ls='-')
                    ax.vlines(x=ave_list[i] + (sigma_list[i] / np.sqrt(len(data[i]))), ymin=y0 - 0.005, ymax=y0 + 0.005,
                              color=color[i], ls='-')
                    ax.hlines(y=0.1, xmin=ave_list[i] - (sigma_list[i] / np.sqrt(len(data[i]))),
                              xmax=mu_list[i] + (sigma_list[i] / np.sqrt(len(data[i]))), color=color[i], ls='-')
            else:
                ax.vlines(x=np.mean(data[i]), ymin=0, ymax=height, color=color[i], ls='--')
                # plot the standard error
                y0 = 0.1
                ax.vlines(x=ave_list[i] - (sigma_list[i] / np.sqrt(len(data[i]))), ymin=y0 - 0.005, ymax=y0 + 0.005,
                          color=color[i], ls='-')
                ax.vlines(x=ave_list[i] + (sigma_list[i] / np.sqrt(len(data[i]))), ymin=y0 - 0.005, ymax=y0 + 0.005,
                          color=color[i], ls='-')
                ax.hlines(y=0.1, xmin=ave_list[i] - (sigma_list[i] / np.sqrt(len(data[i]))),
                          xmax=mu_list[i] + (sigma_list[i] / np.sqrt(len(data[i]))), color=color[i], ls='-')


            print(f'mean value: {np.mean(data[i])}')
    return n_out, ave_list, mu_list, sigma_list, height



