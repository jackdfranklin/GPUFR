import sympy
from sympy import isprime, factorint
import random
import numpy as np
from mod import Mod

def find_good_prime(start, power_of_2, number_of_primes):
    # Define the modulus condition
    mod_value = 2 ** power_of_2
    primes = []
    
    # Start searching for primes from 'start'
    p = start
    number_found = 0;
    while number_found < number_of_primes:
        # Find the next prime >= start
        p = sympy.nextprime(p)
        if (p >= 2147483647):
            print("Ran out of numbers")
            break;
        
        if (p - 1) % mod_value == 0:
            primes.append(p)
            number_found += 1
            print(number_found)
    
    return primes

def find_primitive_root(n, p):
    """ Find a primitive nth root of unity modulo prime p """
    while True:
        x = Mod(random.randint(1, p - 1), p)
        g = x ** ((p-1) // n) #python int division with //
        if g ** (n//2) != 1:
            return g

def primes_roots(start, two_exp, number):
    """
    Finds prime fields with a 2^two_exp principal root and finds principal roots
    from 2^1 to 2^two_exp
    """
    mod_value = 2 ** two_exp
    primes_roots = np.empty([number, two_exp+1])
    
    # Start searching for primes from 'start'
    p = start
    number_found = 0;
    while number_found < number:
        # Find the next prime >= start
        p = sympy.nextprime(p)
        if (p >= 2147483647):
            print("Ran out of numbers")
            break;
        
        if (p - 1) % mod_value == 0:
            for i in range(two_exp):
                primes_roots[number_found][i+1] = find_primitive_root(2**(i+1), p)
            primes_roots[number_found][0] = p
            number_found += 1
            print(number_found, primes_roots[number_found-1])
    
    return primes_roots

# Example: Finding a prime where p-1 is divisible by 2^10
start_prime = 10**9  # Start search around 100 million
power_of_2 = 13     # We want p-1 divisible by 2^10 (n = 1024)
# good_prime = find_good_prime(start_prime, power_of_2, 500)
# np_primes = np.array(good_prime)
primes_and_roots = primes_roots(start_prime, power_of_2, 500)
print(primes_and_roots)
np.savetxt("primes_roots_13.csv", primes_and_roots, fmt='%i')
# np.savetxt("primes_13.csv", np_primes, fmt='%i')