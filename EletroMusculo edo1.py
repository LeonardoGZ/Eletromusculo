import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import math

# Parâmetros do sistema

V = 0.336      # Tensão(Watts)
r = 0.336   # Resistência do fio (ohms)
c = 157.08  # Comprimento do fio (m)
L = 0.1     # Indutância da bobina (henrys)
N = 1000    # Número de espiras da bobina
g = 0.0025  # Comprimento do entreferro (m)
A = 0.00196349540849362077403915211455   # Área de secção transversal do núcleo (m^2)
mu0 = 4 * np.pi * 1e-7  # Permeabilidade magnética do vácuo (H/m)
mur = 2500  # Permeabilidade relativa do material do núcleo
m =  0.002454     # Massa da barra de ferro (kg)
x0 = -0.0025  # Posição inicial da barra de ferro (m)
v0 = 0      # Velocidade inicial da barra de ferro (m/s)
t0 = 0

# Condições iniciais
I0 = 0 
I = I0 + 1      # Corrente elétrica inicial na bobina (A)
B0 = 0
B02 = 0


# Função que calcula a densidade de fluxo magnético no entreferro em função da corrente elétrica na bobina e da posição da barra de ferro
def calc_B(I, x):
    B = mu0 * N * I / g
    if x >= 0.0:
        return B
    else:
        return 0
    
def calc_B1(I, g):
    B1 = mu0 * N * I / g**2
    if g.all() >= 0.000000:
        return B1

def calc_F(B, g):
    F = (B**2 * A * N**2) / (2 * mu0 * g**2 * mur) * (mur + 1)
    if g >= 0.000000:
        return F
    else:
        return 0

B = calc_B(I, x = True)  
B1 = (mu0 * N * I) / g**2
B1 = calc_B(I, g)   
F = calc_F(B,g)
y01 = np.array([B, F])  # Vetor de condições iniciais
y02 = np.array([B0, B02])  # Vetor de condições iniciais

# Função que retorna as derivadas das variáveis de estado
def dydt(g, y):
    B , F = y

    F = calc_F(B,g)
    # Derivada da posição da barra de ferro
    dBdg  = (mu0 * N * I) / g**2
    
    # Derivada da velocidade da barra de ferro
    dFdg = (B**2 * A * N**2) / (2 * mu0 * g**2 * mur) * (mur + 1)
    
    return [dBdg, dFdg]


def dIdt(I1, y1):
   

    dBdi = mu0 * N * I1 / g

    dBdi2 = (mu0 * N * (I1**2)) /(2 * g)

    return [dBdi, dBdi2]

# Solução numérica das EDOs
xf = -0.000001
sol = solve_ivp(dydt, [x0, xf], y01, t_eval=np.linspace(x0, xf, 6))

sol1 = solve_ivp(dIdt, [I0, I], y02, t_eval=np.linspace(I0, I, 10))


u = B1**2
print(u)
print(sol.y[0])
print(sol.y[1])


print(sol1.y[0])
print(sol1.y[1])


# Plotagem dos resultados

plt.figure(figsize=(12, 8))
plt.plot(sol.t, sol.y[0], label='Densidade de Fluxo', color = 'b')
plt.xlabel('Entreferro (M)')
plt.ylabel('Densidade de Fluxo (B)')
plt.legend()
plt.show()

plt.figure(figsize=(12, 8))
plt.plot(sol.t, sol.y[1], label='Força magnética', color = 'g')
plt.xlabel('Entreferro (M)')
plt.ylabel('Força (N)')
plt.legend()
plt.show()


plt.figure(figsize=(12, 8))
plt.plot(sol1.t, sol1.y[0], label='Densidade de Fluxo', color = 'b')
plt.xlabel('Corrente (A)')
plt.ylabel('Densidade de Fluxo (B)')
plt.legend()
plt.show()

plt.figure(figsize=(12, 8))
plt.plot(sol1.t, sol1.y[1], label='Densidade de Fluxo derivado', color = 'b')
plt.xlabel('Corrente (A)')
plt.ylabel('Densidade de Fluxo (B)')
plt.legend()
plt.show()