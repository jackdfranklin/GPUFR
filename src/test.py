import numpy as np
from sympy import ntt 

def indices(i):
    return i + np.floor(np.log2(i+1))

def branch_one(i):
    return i%4 + np.floor(i/4) + 1

def ntt_test():
    # sequence  
    seq = [1, 2, 1, 4] 
    
    prime_no = 105097513
    
    # ntt 
    transform = ntt(seq, prime_no) 
    print ("NTT : ", transform) 

if __name__  == "__main__":
    # iss = np.arange(0, 10, 1)
    # print(indices(iss))
    # print(branch_one(iss))
    ntt_test()