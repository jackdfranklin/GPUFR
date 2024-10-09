import numpy as np
from sympy import ntt 

def indices(i):
    return i + np.floor(np.log2(i+1))

def branch_one(i):
    return i%4 + np.floor(i/4) + 1

def ntt_test():
    # sequence  
    seq = [1, 2, 3, 4] 
    
    prime_no = 1000071169
    
    # ntt 
    transform = ntt(seq, prime_no) # Note sympy.ntt seems to get the output coefficients in the wrong orders
    print ("NTT : ", transform) 
    
def bit_rev(number, size_log2):
    rev_index = 0;
    i = number
    for j in range(size_log2):
        rev_index <<= 1;
        rev_index |= (i & 1);
        i >>= 1;
    
    return rev_index

if __name__  == "__main__":
    # iss = np.arange(0, 10, 1)
    # print(indices(iss))
    # print(branch_one(iss))
    ntt_test()
    # print(bit_rev(7, 3))
    
