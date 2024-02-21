'''EDP de Poisson '''
import numpy as np

a = input("a: ")
b = input("b: ")
c = input("c: ")
d = input("d: ")
N = input("Ingrese el número de N: ")
M = input("Ingrese el número de M: ")
h = (b-a)/N
k = (d-c)/M
w = [[]]

for i in range(N):
    for j in range(M):
        w[i][j] = 0

def f(i, j):
    return 0

#Estos son los puntos de frontera
for i in range(1, N):
    w[i][0] = input("Ingrese el valor de la frontera inferior: ")
    w[i][N] = input("Ingrese el valor de la frontera superior: ")

for j in range(1, N):
    w[0][j] = input("Ingrese el valor de la frontera izquierda: ")
    w[N][j] = input("Ingrese el valor de la frontera derecha: ")

#Estos son los puntos interiores
for k in range(100):
    for i in range(1, N):
        for j in range(1, M):
            w[i][j] = (k**2*(w[i+1][j]+w[i-1][j]) + h**2*(w[i][j+1]+w[i][j-1]) + f(i, j)*h**2*k**2)/(2*(h**2+k**2))

