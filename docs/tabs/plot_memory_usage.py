# -*- coding: utf-8 -*-
"""
This script plots the memory usage of SPOD versus the number of data

Xiao He (xiao.he2014@imperial.ac.uk),
Last update: 24-Sep-2021
"""

# -------------------------------------------------------------------------
# utilities
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pylab

params={
'axes.labelsize': '20',       
'xtick.labelsize': '16',
'ytick.labelsize': '16',
'lines.linewidth': 1.5,
'legend.fontsize': '14',
'figure.figsize': '8, 6'    # set figure size
}
pylab.rcParams.update(params)
    
def figure_format(xtitle, ytitle, zoom, legend):
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.axis(zoom)
    plt.grid(linestyle='dashed')
    if legend != 'None':
        plt.legend(loc=legend)

# -------------------------------------------------------------------------
# load data
data = pd.read_csv('memory_usage.csv')

# linear fit of data
x = data['Nt']*data['Nq']
x_fit = 10**np.linspace(np.log10(x.min()), np.log10(x.max()), 10)
x = np.vstack([x, np.ones(len(x))]).T
a_fast, b_fast = np.linalg.lstsq(x, data['Memory_fast'].values, rcond=None)[0]
a_lowRAM, b_lowRAM = np.linalg.lstsq(x, data['Memory_lowRAM'].values, rcond=None)[0]

# begin plot
plt.figure()
plt.scatter(data['Nt']*data['Nq'], data['Memory_fast'], marker='o', color='indianred', label='fast')
plt.scatter(data['Nt']*data['Nq'], data['Memory_lowRAM'], marker='v', color='steelblue', label='lowRAM')
plt.plot(x_fit, a_fast*x_fit+b_fast, label='', linestyle='--', color='indianred')
plt.plot(x_fit, a_lowRAM*x_fit+b_lowRAM, label='', linestyle=':', color='steelblue')
plt.text(3*10**6,2*10**1,r'$y=%.4f \times 10^{-8} x + %.2f$'%(a_fast*10**8,b_fast), 
         fontsize=14, color='indianred')
plt.text(2*10**8,3*10**(-1),r'$y=%.4f \times 10^{-8} x + %.2f$'%(a_lowRAM*10**8,b_lowRAM), 
         fontsize=14, color='steelblue')
figure_format(r'$N_t \cdot N_q$', 'Memoray usage (GB)', [10**6,10**11,10**(-1),10**3],'best')
plt.xscale('log')
plt.yscale('log')
plt.savefig('../figs/SPOD_memory_usage.png', dpi=300)