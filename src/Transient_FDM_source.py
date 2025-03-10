"""
@author: brand
"""
import matplotlib.pyplot as plt
import numpy as np

def solinsta(Deff, k, Ce, Ro, n, dt, t_final):    
    dr = Ro / (n - 1)  # Pas spatial [m]
    
    nt = t_final//dt + 1
    nt=int(nt)
    
    # Vecteur des positions radiales
    r = np.linspace(0, Ro, n)
    
    # Initialisation de la concentration
    C = np.zeros((nt,n))  
    
    # Construction de la matrice du schéma implicite
    A = np.zeros((n, n))  # Matrice des coefficients
    B = np.zeros(n)        # Second membre
    
    # Remplissage de la matrice pour les points intérieurs
    for i in range(1, n-1):
        alpha = Deff * dt / dr**2
        beta = Deff * dt / (2 * dr * r[i])
    
        A[i, i-1] = alpha - beta
        A[i, i] = -2*alpha -k*dt - 1
        A[i, i+1] = alpha + beta
    
    # Condition aux limites : Neumann (symétrie) en r=0
    A[0, 0] = -3 / (2 * dr)
    A[0, 1] = 4 / (2 * dr)
    A[0, 2] = -1 / (2 * dr)
    B[0] = 0  # dC/dr = 0 à r=0
    
    # Condition aux limites : Dirichlet en r=Ro (concentration fixée)
    A[-1, -1] = 1
    
    # Simulation temporelle
    t_array=[0]
    t=0
    for n_t in range(1, nt):
        # Pour les points intérieurs : on utilise la solution de l'instant précédent
        for i in range(1, n-1):
            B[i] = - C[n_t - 1, i]
        # Pour le bord en r=0, on impose dC/dr = 0
        B[0] = 0
        # Pour le bord en r=Ro, on impose la valeur Dirichlet Ce
        B[-1] = Ce
        
        # Résolution du système A * C_new = B
        C[n_t, :] = np.linalg.solve(A, B)
        t+=dt
        t_array.append(t)
        
    return(C,r,nt, t_array)

# Paramètres physiques
Deff = 1e-10   # Coefficient de diffusion effectif [m²/s]
k = 4e-9       # Constante de réaction [1/s]
Ce = 20        # Concentration à la surface [mol/m³]
Ro = 0.5       # Rayon externe du pilier [m]
n = 11       # Nombre de points en espace
dt = 200000    # Pas de temps [s]
t_final = 4e9 

C,r,nt,t_array= solinsta(Deff, k, Ce, Ro, n, dt, t_final)


plt.figure(figsize=(8, 6),dpi=150)
for i in range(0,41,4):
    indice = i**2 
    temps=t_array[indice]
    plt.plot(r, C[indice, :], '.-', label=f"$t ={temps:.2e}$ ")
plt.plot(r, C[nt-1, :], '.-', label=f"$t ={t_final:.2e}$ ")
plt.xlabel("Rayon [m]")
plt.ylabel("Concentration [mol/m³]")
plt.title("Profils de concentration")
plt.grid()
plt.legend()
plt.show()
