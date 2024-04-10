import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

'''
Derivada segunda:
u_xx = (w[i+1][j] - 2*w[i][j] + w[i-1][j]) / h**2
u_yy = (w[i][j+1] - 2*w[i][j] + w[i][j-1]) / k**2

Derivada cruzada:
u_xy = (w[i+1][j+1] + w[i-1][j-1] - w[i-1][j+1] - w[i+1][j-1]) / (4*h*k)

Derivada simple de x:
    - Progresivas: (w[i+1][j] - w[i][j]) / h
    - Regresivas: (w[i][j] - w[i-1][j]) / h

Derivada simple de y:
    - Progresiva: (w[i][j+1] - w[i][j]) / k
    - Regresiva: (w[i][j] - w[i][j-1]) / k
'''

# Entrada de parámetros
N = int(input('Ingresar valor de N: '))
M = int(input('Ingresar valor de M: '))
a = float(input('Ingresar valor de a: '))
b = float(input('Ingresar valor de b: '))
c = float(input('Ingresar valor de c: '))
d = float(input('Ingresar valor de d: '))
h = (b-a) / N
k = (d-c) / M


# Inicialización de la matriz w con dimensiones correctas
w = np.zeros((N+1, M+1))  # +1 para incluir los bordes

#Función f(i, j)
#Cuando tenemos una funcion en varias partes
'''def f(x):
    if 0 < x < b/2:
        return 1
    else:
        return 0'''

def f(x):
    #return np.exp(-(h*i-2.5)**2)
    return np.exp(-((h*i-0.25)**2))    #u(x,0)

for i in range(N):
    w[i][0] = f(i*h)  # Frontera inferior
    #w[i][M] = 0  Frontera superior, recordar que x_i= x_o(a) + ih
   
for j in range(M):
    w[0][j] = 0  # Frontera izquierda, recordar que t_i = t_o + jk   u(n,t)
    w[N][j] = 0  # Frontera derecha u(0, t)

''' Para que la onda no sea estacionaria 
    w[0][j] = 3*np.sin(k*j) #Frontera izquierda, recordar que t_i = jk
    w[N][j] = np.sin(k*j)  # Frontera derecha
'''

# Iteraciones para la solución
for iter in range(100):  # Número de iteraciones
    for j in range(1, M):
        for i in range(1, N):
            #w[i][j] = (k*(w[i+1][j]+w[i-1][j]) + (h**2)*(1+h*i)*w[i][j-1]) / (2*k+(h**2)*(1+h*i))
            #w[i][j] = ((k/(h**2))*(w[i+1][j]+w[i-1][j]) + (1+h*i)*w[i][j-1]) / (1+h*i+2*(k/(h**2)))
            #w[i][j] = (k*(w[i+1][j]+w[i-1][j]) + (1+h*i)*w[i][j-1]*h**2) / (2*k+(1+h*i)*h**2-(h**2)*k)
            #w[i][j] = (k*(w[i+1][j]+w[i-1][j]) + (1+h*i)*w[i][j-1]*h**2) / (2*k+(1+h*i)*h**2-(h**2)*k*(h*i))
            #w[i][j] = (2*(k**2)*(w[i+1][j]+w[i-1][j]) + (h**2)*2*(w[i][j+1]+w[i][j-1]) + h*k*(w[i+1][j+1]+w[i-1][j-1]-w[i-1][j+1]-w[i+1][j-1])) / (8*k**2)
            #w[i][j] = (1/2)*(w[i+1][j]+w[i-1][j]) + (h/(8*k))*(w[i+1][j+1]+w[i-1][j-1] - w[i-1][j+1] - w[i+1][j-1])
            w[i][j] = (4*(k**4)*(w[i+1][j]+w[i-1][j]) + (k**3)*h*(w[i+1][j+1]+w[i-1][j-1]-w[i-1][j+1]-w[i+1][j-1]) + 4*(h**2)*(w[i][j+1]+w[i][j-1])) / (8*(k**2)*((k**2)+(h**2)))

'''40, 400 para las parabólicas, 30 30 para las elipticas'''

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
plt.show()