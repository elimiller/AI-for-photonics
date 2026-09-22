# %% Imports 
import numpy as np
import matplotlib.pyplot as plt
# %% Define function
def f(x):
    return x*x
# %% Plot function
def plotf(f,x):
    fig = plt.plot(x,f(x))
    return fig

# %% Run code
x = np.linspace(0,1,10)

plotf(f,x)
plt.show()
# %%
