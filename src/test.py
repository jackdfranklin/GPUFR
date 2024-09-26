import numpy as np

def indices(i):
    return i + np.floor(np.log2(i+1))

def branch_one(i):
    return i%4 + np.floor(i/4) + 1

if __name__  == "__main__":
    iss = np.arange(0, 10, 1)
    print(indices(iss))
    print(branch_one(iss))