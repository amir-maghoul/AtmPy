# main.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def ellipric(x):
    return x**2

def main():
    x = np.linspace(-5, 5, 5)
    f = ellipric(x)
    plt.plot(x, f, "-o")
    plt.grid()
    plt.show()



if __name__ == "__main__":
    main()
