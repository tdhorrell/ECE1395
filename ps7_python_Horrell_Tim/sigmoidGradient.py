#sigmoidGradient.py
from sigmoid import sigmoid

def sigmoidGradient(z):
    g_prime = []
    for val in z:
        g_prime.append(sigmoid(val) * (1 - sigmoid(val)))

    return g_prime