import numpy as np
import matplotlib.pyplot as plt

# Define as constantes do problema
N = 100  # Número de espiras da bobina
I = 1.0  # Corrente elétrica na bobina
g = 0.0025  # Comprimento do entreferro
mu_r = 2500  # Permeabilidade relativa do material do núcleo
mu_0 = 4 * np.pi * 10**(-7)  # Permeabilidade magnética do vácuo
A = 0.00196349540849362077403915211455  # Área de seção transversal do núcleo

# Define as variáveis do problema
t = 0  # Tempo inicial
dt = 0.01  # Passo de tempo
B = mu_0 * N * I / g  # Densidade de fluxo magnético inicial
F = (B**2 * A * N**2) / (2 * mu_0 * g * mu_r) * (mu_r + 1)  # Força inicial
x = [t]  # Vetor de tempo
y1 = [B]  # Vetor de densidade de fluxo magnético
y2 = [F]  # Vetor de força

# Loop de simulação
while t < 10:
    t += dt
    B = mu_0 * N * I / (g + 0.001 * np.sin(2 * np.pi * t))  # Calcula a densidade de fluxo magnético atual
    F = (B**2 * A * N**2) / (2 * mu_0 * (g + 0.001 * np.sin(2 * np.pi * t)) * mu_r) * (mu_r + 1)  # Calcula a força atual
    x.append(t)  # Adiciona o tempo atual ao vetor de tempo
    y1.append(B)  # Adiciona a densidade de fluxo magnético atual ao vetor correspondente
    y2.append(F)  # Adiciona a força atual ao vetor correspondente

    # Plota os gráficos em tempo real
    plt.clf()
    plt.subplot(2, 1, 1)
    plt.plot(x, y1)
    plt.title('Densidade de fluxo magnético em função do tempo')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Densidade de fluxo magnético (T)')
    plt.subplot(2, 1, 2)
    plt.plot(x, y2)
    plt.title('Força em função do tempo')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Força (N)')
    plt.pause(0.001)

plt.show()
