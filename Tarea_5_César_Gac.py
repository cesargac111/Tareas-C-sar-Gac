from posixpath import altsep
import numpy as np
from scipy.optimize import bisect

# Función para calcular Z y Varianza
def moments(lmbda, M):
    xs = np.arange(-M, M+1)
    Alpha = np.exp(-lmbda * xs**2)
    Z = np.sum(Alpha)
    var = np.sum(xs**2 * Alpha) / Z
    return Z, var

# Dada una varianza sigma2, encontrar lambda
def find_lambda(sigma2, M=50):
    # función cuya raíz buscamos: Var(lambda) - sigma2 = 0
    f = lambda lmbda: moments(lmbda, M)[1] - sigma2
    # buscamos raíz en [1e-6, 10]
    lmbda = bisect(f, 1e-6, 10)
    return lmbda

# Construimos la distribución máxima entropía
def maxent_distribution(sigma2, M=50):
    lmbda = find_lambda(sigma2, M)
    xs = np.arange(-M, M+1)
    Alpha = np.exp(-lmbda * xs**2)
    Z = np.sum(Alpha)
    ps = Alpha / Z
    return xs, ps, lmbda

# Ejemplo
if __name__ == "__main__":
    sigma2 = 10  # varianza deseada
    xs, ps, lmbda = maxent_distribution(sigma2)

    print(f"Lambda encontrado: {lmbda:.5f}")
    print("Distribución de probabilidad (p(x)):")
    for x, p in zip(xs, ps):
        if p > 1e-4:
            print(f"x={x:3d}, p={p:.5f}")

# Nota: 4.0
# No se entiende por qué la exponencial usada
# maximiza la entropía. Faltó argumentarlo.
# Esta derivación era la parte central del
# ejercicio.
# El cero se podía encontrar analíticamente, no 
# era necesario usar la bisección.
