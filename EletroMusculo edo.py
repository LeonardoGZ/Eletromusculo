import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Parâmetros do sistema
V = 12      # Tensão aplicada na bobina (V)
R = 10      # Resistência elétrica da bobina (ohms)
L = 0.1     # Indutância da bobina (henrys)
N = 1000    # Número de espiras da bobina
g = 0.0025  # Comprimento do entreferro (m)
A = 0.00196349540849362077403915211455   # Área de secção transversal do núcleo (m^2)
mu0 = 4 * np.pi * 1e-7  # Permeabilidade magnética do vácuo (H/m)
mur = 2500  # Permeabilidade relativa do material do núcleo
m = 0.1     # Massa da barra de ferro (kg)
x0 = 0.01   # Posição inicial da barra de ferro (m)
v0 = 0      # Velocidade inicial da barra de ferro (m/s)

# Condições iniciais
I0 = 1      # Corrente elétrica inicial na bobina (A)
t0 = 0      # Tempo inicial (s)
y0 = np.array([I0, x0, v0])  # Vetor de condições iniciais

# Função que calcula a densidade de fluxo magnético no entreferro em função da corrente elétrica na bobina e da posição da barra de ferro
def calc_B(I, x):
    B = mu0 * N * I / g
    if x >= 0:
        return B
    else:
        return 0

# Função que retorna as derivadas das variáveis de estado
def dydt(t, y):
    I, x, v = y
    
    # Calcula a força magnética no entreferro
    B = calc_B(I, x)
    F = (B**2 * A * N**2) / (2 * mu0 * g * mur) * (mur + 1)
    
    # Derivada da corrente elétrica na bobina
    dIdt = (V - I * R - F * N) / L
    
    # Derivada da posição da barra de ferro
    dxdt = v
    
    # Derivada da velocidade da barra de ferro
    dvdt = F / m
    
    return [dIdt, dxdt, dvdt]

# Solução numérica das EDOs
tf = .5
sol = solve_ivp(dydt, [t0, tf], y0, t_eval=np.linspace(t0, tf, 1000))

    # Plotagem dos resultados

    
plt.figure(figsize=(12, 6))
plt.plot(sol.t, sol.y[0], label='Corrente elétrica')
plt.plot(sol.t, sol.y[1], label='Posição da barra de ferro')
plt.plot(sol.t, sol.y[2], label='Velocidade da barra de ferro')
plt.xlabel('Tempo (s)')
plt.legend()
plt.show()
