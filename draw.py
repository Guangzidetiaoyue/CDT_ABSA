import os

import matplotlib.pyplot as plt

import numpy as np


def draw_curve(train_log, test_log, curve_type, epoch):
    x = list(range(epoch))
    
    # show acc
    plt.plot(x, train_log, c="blue", label="train")
    plt.plot(x, test_log, c="orange", label="test")
    plt.xlabel("epoch")
    plt.ylabel(curve_type)
    if curve_type == "acc":
        plt.ylim((0, 100))
    else:
        plt.ylim((0, 4))
    plt.legend()
    plt.show()
