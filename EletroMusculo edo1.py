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
m =  0.002454     # Massa da barra de ferro (kg)
x0 = -0.0025  # Posição inicial da barra de ferro (m)
v0 = 0      # Velocidade inicial da barra de ferro (m/s)

# Condições iniciais
I0 = 1      # Corrente elétrica inicial na bobina (A)
t0 = -0.0025     # Tempo inicial (s)
y0 = np.array([I0, x0, v0])  # Vetor de condições iniciais

B0 = 0.502
F0 = 0


# Função que calcula a densidade de fluxo magnético no entreferro em função da corrente elétrica na bobina e da posição da barra de ferro
def calc_B(I, x):
    B = mu0 * N * I / g
    if x.all() <= 0.0:
        return B
    else:
        return 0
    
def calc_B1(I, s):
    B1 = mu0 * N * I / g**2
    if s.all() >= 0.000000:
        return B1
    else:
        return 0
pot = 2

Bp = B0**pot
def calc_F(Bp, g):
    F = (Bp * A * N**2) / (2 * mu0 * g * mur) * (mur + 1)
    if g >= 0.000000:
        return F
    else:
        return 0
    
F = calc_F(B0,g)
y01 = np.array([B0, F, v0])  # Vetor de condições iniciais


# Função que retorna as derivadas das variáveis de estado
def dydt(t, y):
    I, x, s1 = y
 
    # Calcula a força magnética no entreferro
    B = calc_B(I, x)
    #B1=calc_B1(I, s)
 
    F = calc_F(Bp, s1)
    
    # Derivada da corrente elétrica na bobina (que é constante)
    dIdt = 1
    
    # Derivada da posição da barra de ferro
    dBdg  = (mu0 * N * I) / g**2
    
    # Derivada da velocidade da barra de ferro
    dFdg = (Bp * A * N**2) / (2 * mu0 * g**2 * mur) * (mur + 1)
    
    return [dIdt, dBdg, dFdg]



# Solução numérica das EDOs
tf = 0.0
sol = solve_ivp(dydt, [t0, tf], y01, t_eval=np.linspace(t0, tf))




#sol = solve_ivp(dy1dt, [x0, xf], y01, t_eval=np.linspace(x0, xf, 25))

# Plotagem dos resultados
plt.figure(figsize=(12, 8))
#plt.plot(sol.t, sol.y[0], label='Corrente elétrica' , color='red')
plt.plot(sol.t, sol.y[1], label='Densidade de Fluxo')
plt.plot(sol.t, sol.y[2], label='Força magnética')
plt.xlabel('Entreferro (M)')
plt.ylabel('Força (N)')
plt.legend()
plt.show()



