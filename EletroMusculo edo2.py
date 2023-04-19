import numpy as np
import matplotlib.pyplot as plt

# Parâmetros do sistema
N = 1000    # Número de espiras da bobina
A = 0.00196349540849362077403915211455   # Área de secção transversal do núcleo (m^2)
mu0 = 4 * np.pi * 1e-7  # Permeabilidade magnética do vácuo (H/m)
mur = 2500  # Permeabilidade relativa do material do núcleo

# Cria um array com valores para o entreferro
g_array = np.linspace(-0.0025, -0.0000001, 10)

B= (mu0*N*1) / g_array**2

print(B)

F= 174.38, 174.38, 174.38, 174.38, 174.38, 174.38, 174.38, 174.38, 174.38, 174.38 
y = -0.0025, -0.0025,-0.0025,-0.0025,-0.0025,-0.0025,-0.0025,-0.0025,-0.0025,-0.0025
# Calcula a derivada da força magnética em relação ao entreferro para cada valor de g
dFdg = (B**2 * A * N**2) / (2 * mu0 * g_array**2 * mur) * (mur + 1)

# Plota o gráfico da derivada da força magnética em relação ao entreferro
plt.plot(g_array, dFdg)
plt.plot(y, F, color='black', linewidth=20)
plt.xlabel('Entreferro (m)')
plt.ylabel('Derivada da força magnética em relação ao entreferro (N/m)')
plt.title('Derivada da força magnética em relação ao entreferro')
plt.show()

plt.plot(g_array, B, color = 'red')
plt.xlabel('Entreferro (m)')
plt.ylabel('Derivada da densidade de fluxo magnético em relação ao entreferro (N/m)')
plt.title('Derivada da densidade de fluxo magnético em relação ao entreferro')

plt.show()