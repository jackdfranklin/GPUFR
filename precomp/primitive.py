from sympy import isprime, factorint
import random

def find_primitive_root(n, p):
    """ Find a primitive nth root of unity modulo prime p """
    if not isprime(p):
        raise ValueError(f"{p} is not a prime number.")

    # Ensure that p - 1 is divisible by n
    if (p - 1) % n != 0:
        raise ValueError(f"{p} - 1 is not divisible by {n}, so no nth root of unity exists.")

    # Factor p-1 to get divisors (orders of elements in the multiplicative group)
    factors = factorint(p - 1)

    # To check if g is a primitive nth root of unity, g^n should be 1 mod p and 
    # no smaller power of g should be 1 mod p.
    def is_primitive_root(g, n, p):
        if pow(g, n, p) != 1:
            return False
        # Check that g^k != 1 mod p for all k < n
        for k in range(1, n):
            if pow(g, k, p) == 1:
                return False
        return True

    # Try random elements until a primitive nth root of unity is found
    while True:
        g = random.randint(2, p - 2)  # Randomly select g from 2 to p-2
        if is_primitive_root(g, n, p):
            return g

# Example usage
n = 2048
p = 1004535809  # Prime number such that p - 1 is divisible by n

try:
    root_of_unity = find_primitive_root(n, p)
    print(f"The primitive {n}-th root of unity modulo {p} is {root_of_unity}.")
except ValueError as e:
    print(e)
