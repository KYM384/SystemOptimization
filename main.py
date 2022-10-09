from matplotlib import pyplot as plt
import numpy as np
import functools

import methods
import utils


def f(x):
    x1, x2 = x[:,0]
    return (x1-2)**4 + (x1-2*x2)**2

def g(x):
    x1, x2 = x[:,0]
    return x1**2 - x2

x_init = np.array([1, 2]).reshape(-1, 1)
_, x_history = methods.SQP(f, [g], x_init, solver="active")

print(len(x_history))
print(x_history[-1].reshape(-1))

utils.visualize(f, x_history, [g])
plt.savefig("visualize.png")