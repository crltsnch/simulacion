import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Entrada de parámetros
N = int(input('Ingresar valor de N: '))
M = int(input('Ingresar valor de M: '))
b = float(input('Ingresar valor de b: '))
d = float(input('Ingresar valor de d: '))
h = b / N
k = d / M
vel_max = h/k
print('Velocidad máxima: ', vel_max)
v = float(input('velocidad: '))

# Inicialización de la matriz w con dimensiones correctas
w = np.zeros((N+1, M+1))  # +1 para incluir los bordes

# Función f(i, j)
'''Cuando tenemos una funcion en varias partes
def f(x):
    if x < b/2:
        return x
    else:
        return b-x'''

def f(x):
    if 1<x<3:
        return 1
    else:
        return 0

# Función g(i) para la condición inicial
def g(x):
    return 0

for i in range(N):
    w[i][0] = f(i*h)  # Frontera inferior
    w[i][1] = w[i][0] + k*g(i*h)
   

for j in range(M):
    w[0][j] =0 #Frontera izquierda, recordar que t_i = jk
    w[N][j] = 0  # Frontera derecha
''' Para que la onda no sea estacionaria 
    w[0][j] = 3*np.sin(k*j) #Frontera izquierda, recordar que t_i = jk
    w[N][j] = np.sin(k*j)  # Frontera derecha
'''
# Iteraciones para la solución
for j in range(1, M):
    for i in range(1, N):
        w[i][j+1] = 2*(1-(v*k/h)**2)*w[i][j] + ((v*k/h)**2)*(w[i+1][j]+w[i-1][j]) - w[i][j-1]

# Crear una malla de coordenadas para graficar
x = np.linspace(0, b, N+1)
y = np.linspace(0, d, M+1)
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
plt.show()