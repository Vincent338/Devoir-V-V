import os
import matplotlib.pyplot as plt
import numpy as np

def FDM_2ordre(prm, save_plots=False):
    """Fonction simulant avec la méthode des différences finies d'ordre 2

    Entrées:
        - prm : Objet class parametres()
            - Deff : Coefficient de diffusion effectif du sel dans le béton poreux
            - S : Température ambiante autour de la conduite [mol/m^3/s]
            - Ce : Concentration à la surface du pilier [mol/m^3]
            - Ri : Rayon interne [m]
            - Ro : Rayon externe [m]
            - n : Nombre de noeuds [-]
            - dr : Pas en espace [m]

    Sortie (dans l'ordre énuméré ci-bas):
        - Vecteur R (array) de dimension n composé de la position radiale à laquelle les concentrations sont calculées, où n le nombre de noeuds.
        - Vecteur C (array) de dimension n composé de la concentration en fonction du rayon, où n le nombre de noeuds
    """

    Deff = prm.Deff
    k = prm.k
    Ce = prm.Ce
    Ri = prm.Ri
    Ro = prm.Ro
    n = prm.n
    dr = prm.dr
    dt = prm.dt
    tf = prm.tf
    
    t0 = 0
    t = [0]
    
    R = np.empty(n)
    
    M = np.zeros((n,n))
    B = np.zeros(n)
    
    for i in range(0,n):
        R[i] = Ri + i*dr
    
    for i in range(0,n):
        
        if i == 0:
            M[i, i] = -3
            M[i, i+1] = 4
            M[i, i+2] = -1
            
        elif i == n-1:
            M[i, i] = 1
            
            B[i] = Ce
            
        else:
            M[i, i-1] = (-dt * Deff / (dr**2) + dt * Deff / (2 * R[i] * dr))
            M[i, i]   = (1 + 2 * dt * Deff / (dr**2) + dt * k)
            M[i, i+1] = (-dt * Deff / (dr**2) - dt * Deff / (2 * R[i] * dr))
            
    
    print(M)
    
    while (t0 <= tf) == True:
        
        "Résolution du système matriciel"        
        C = np.linalg.solve(M, B)
                    
        B = C.copy()
        B[0] = 0
        t0 = t0 + dt
        t.append(t0)
                
    return C, R, t

def Sol_anal(r, prm):
    
    "Calcul de la solution analytique selon la position radiale sur le domaine r"
    
    Deff = prm.Deff
    S = prm.S
    Ce = prm.Ce
    Ro = prm.Ro
    
    Sol = 0.25*(S/Deff)*(Ro**2)*(r**2/Ro**2 - 1) + Ce
    
    return Sol

class Parametres:
    def __init__(self):
        self.D = 1
        self.Deff = 1e-10
        self.k = 4e-9
        self.S = 2e-8
        self.Ce = 20
        self.Ri = 0
        self.Ro = self.D / 2
        self.n = 5 
        self.dr = (self.Ro - self.Ri) / (self.n - 1)
        self.dt = 3e7  
        self.tf = 4e9
    
prm = Parametres()


"--------------------QUESTION 1D)--------------------"

C, R, t = FDM_2ordre(prm, save_plots = False)

td = t[-1] 

plt.figure(1)
plt.scatter(R, C, color='b', marker='o', label="Numérique (FDM de 2e ordre)")
plt.xlabel("Position radiale r dans le pilier (m)")
plt.ylabel("Concentration C (mol/m³)")
plt.title(f"Profil de concentration stationnaire du \n processus de diffusion du sel dans le pilier de béton à {td} s")
plt.legend()
plt.grid()
plt.show()













