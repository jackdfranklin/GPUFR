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

def save_as_cpp_array(array, filename, variable_name):
    """
    Saves a multidimensional NumPy array to a file as a C++ array.
    
    :param array: Multidimensional NumPy array to save.
    :param filename: Name of the output .h file.
    :param variable_name: Name of the C++ variable to create.
    """
    with open(filename, "w") as file:
        # Write header guard and includes
        file.write("#pragma once\n\n")
        file.write("#include \"GPUFR/types.hpp\"\n\n")
        
        # Write the array as a C++ array
        file.write(f"const u32 {variable_name}[{array.shape[0]}][{array.shape[1]}] = {{\n")
        for row in array:
            formatted_row = [int(x) for x in row]
            file.write("    { " + ", ".join(map(str, formatted_row)) + " },\n")
        file.write("};\n")

# Example: Finding a prime where p-1 is divisible by 2^10
start_prime = 10**9  # Start search around 100 million
power_of_2 = 15     # We want p-1 divisible by 2^10 (n = 1024) need to double to capture full NTT
# good_prime = find_good_prime(start_prime, power_of_2, 500)
# np_primes = np.array(good_prime)
primes_and_roots = primes_roots(start_prime, power_of_2, 50)
print(primes_and_roots)
# np.savetxt("primes_roots_14.csv", primes_and_roots, fmt='%i')
save_as_cpp_array(primes_and_roots, "../include/GPUFR/precomp.hpp", "precomp")
# np.savetxt("primes_13.csv", np_primes, fmt='%i')