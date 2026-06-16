#%%
import numpy as np



prediction = np.loadtxt('/utrecht_exp/results/prediction_history.npy')
true = np.loadtxt('/utrecht_exp/results/true_history.npy')
import matplotlib.pyplot as plt


plt.plot(true[100:200,1])



