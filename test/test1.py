import numpy as np
import pandas as pd








if __name__ == "__main__":
    # price = np.loadtxt("data/sh_600000.txt", delimiter=',', usecols=(2,))
    # print(price)

    with open("data/sh_600000.txt", "r") as f:
        data = [line.strip().split("\t") for line in f.readlines()]

    data = pd.DataFrame(data, columns=data[0])
    print(data)
    # data = pd.DataFrame(data)

