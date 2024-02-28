import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Entrada de parámetros
N = int(input('Ingresar valor de N: '))
M = int(input('Ingresar valor de M: '))
a = float(input('Ingresar valor de a: '))
b = float(input('Ingresar valor de b: '))
c = float(input('Ingresar valor de c: '))
d = float(input('Ingresar valor de d: '))

# Cálculo de pasos
h = (b - a) / N
k = (d - c) / M
l = float(input('Ingresar valor de l: '))

# Inicialización de la matriz w con dimensiones correctas
w = np.zeros((N+1, M+1))  # +1 para incluir los bordes

# Función f(i, j) como fuente
def f(i, j):
    return 0
    # return 0 ecuacion de poisson
    #return (a+h*i)*(c+k*j)*(1-a-h*i)*(1-c-k*j) 
  
for i in range(N):
    w[i][0] = 0  # Frontera inferior
    w[i][M] =  0  #(a+h*i)**2 Frontera superior, recordar que x_i= x_o(a) + ih

for j in range(M):
    w[0][j] =  0  #1-(c+k*j)**2  Frontera izquierda
    w[N][j] = 1  # Frontera derecha

# Iteraciones para la solución
for iter in range(100):  # Número de iteraciones
    for i in range(1, N):
        for j in range(1, M):
            w[i][j] = (k**2*(w[i+1][j]+w[i-1][j]) + h**2*(w[i][j+1]+w[i][j-1]) - 2*h**2*k**2*f(i,j)) / (2*(h**2 + k**2) + h**2*k**2*(l**2))

# Crear una malla de coordenadas para graficar
x = np.linspace(a, b, N+1)
y = np.linspace(c, d, M+1)
X, Y = np.meshgrid(x, y)

# Crear la figura y el eje para la gráfica 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Graficar la superficie
surf = ax.plot_surface(X, Y, w.T, cmap='viridis', edgecolor='none')  # Transponer w para que coincida con las dimensiones de X y Y
ax.set_title('Superficie 3D de la Matriz w')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('w')

# Añadir barra de colores para la escala
fig.colorbar(surf)
plt.savefig('EDPPoisson1.png')
plt.show()