# -*- coding: utf-8 -*-
"""
@author: brand
"""

import matplotlib.pyplot as plt
import numpy as np

def FDM_1ordre(prm):
    """Fonction simulant avec la méthode des différences finies d'ordre 1

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
    S = prm.S
    Ce = prm.Ce
    Ri = prm.Ri
    Ro = prm.Ro
    n = prm.n
    dr = prm.dr
    
    R = np.empty(n)
    
    for i in range(0,n):
        R[i] = Ri + i*dr
           
    M = np.zeros((n,n))
    B = np.empty(n)
    
    for i in range(0,n):
        
        if i == 0:
            M[i, i] = -1
            M[i, i+1] = 1
            
            B[i] = 0
            
        elif i == n-1:
            M[i, i] = 1
            
            B[i] = Ce
            
        else:
            M[i, i-1] = R[i]/dr**2
            M[i, i]   = -2*R[i]/dr**2 - 1/dr
            M[i, i+1] = R[i]/dr**2 + 1/dr
            
            B[i] = S*R[i]/Deff
    
    "Résolution du système matriciel"  
    C = np.linalg.solve(M, B) 
    
    return R, C

def FDM_2ordre(prm):
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
    S = prm.S
    Ce = prm.Ce
    Ri = prm.Ri
    Ro = prm.Ro
    n = prm.n
    dr = prm.dr
    
    R = np.empty(n)
    
    M = np.zeros((n,n))
    B = np.empty(n)
    
    for i in range(0,n):
        R[i] = Ri + i*dr
    
    for i in range(0,n):
        
        if i == 0:
            M[i, i] = -3
            M[i, i+1] = 4
            M[i, i+2] = -1
            
            B[i] = 0
            
        elif i == n-1:
            M[i, i] = 1
            
            B[i] = Ce
            
        else:
            M[i, i-1] = R[i]/dr**2 - 1/(2*dr)
            M[i, i]   = -2*R[i]/dr**2
            M[i, i+1] = R[i]/dr**2 + 1/(2*dr)
            
            B[i] = S*R[i]/Deff
            
    "Résolution du système matriciel"        
    C = np.linalg.solve(M, B) 
    
    return R, C

def Sol_anal(r, prm):
    
    "Calcul de la solution analytique selon la position radiale sur le domaine r"
    
    Deff = prm.Deff
    S = prm.S
    Ce = prm.Ce
    Ro = prm.Ro
    
    Sol = 0.25*(S/Deff)*(Ro**2)*(r**2/Ro**2 - 1) + Ce
    
    return Sol

class parametres():
    D = 1
    Deff = 10**(-10)
    S = 2 * 10**-8
    Ce = 20
    Ri = 0
    Ro = D/2
    n = 5
    dr = (Ro - Ri)/(n-1)
    
prm = parametres()


"--------------------QUESTION 1D)--------------------"

Sol1 = FDM_1ordre(prm)

r = np.linspace(prm.Ri, prm.Ro, 201)

f_anal = Sol_anal(r, prm)


plt.figure(1)
plt.scatter(Sol1[0], Sol1[1], color='b', marker='o', label="Numérique (FDM de 1er ordre)")
plt.plot(r, f_anal, 'k-', label="Analytique")
plt.xlabel("Position radiale r dans le pilier (m)")
plt.ylabel("Concentration C (mol/m³)")
plt.title("Profil de concentration stationnaire du \n processus de diffusion du sel dans le pilier de béton")
plt.legend()
plt.grid()
custom_yticks = [7.5, 10.625, 12, 14, 16, 18, 20]  # Change these values as needed
plt.yticks(custom_yticks)
plt.savefig('Ordre1.png', dpi=300)
plt.show()


"Vérification de l'erreur selon la FDM de 1er ordre selon 6 maillage différent"
nodes = [5, 10, 20, 50, 75, 100]  
L1_errors, L2_errors, Linf_errors = [], [], []
h_values = []

for n in nodes:
    prm.n = n
    prm.dr = (prm.Ro - prm.Ri) / (n-1)
    R_num, C_num = FDM_1ordre(prm)
    
    C_anal_num_nodes = Sol_anal(R_num, prm)   
    error_L1 = np.mean(np.abs(C_num - C_anal_num_nodes))
    error_L2 = np.sqrt(np.mean((C_num - C_anal_num_nodes)**2))
    error_Linf = np.max(np.abs(C_num - C_anal_num_nodes))

    L1_errors.append(error_L1)
    L2_errors.append(error_L2)
    Linf_errors.append(error_Linf)
    h_values.append(prm.dr)
        
L1_errors = np.array(L1_errors)
L2_errors = np.array(L2_errors)
Linf_errors = np.array(Linf_errors)
    

coeff_L1 = np.polyfit(np.log(h_values), np.log(L1_errors), 1)
coeff_L2 = np.polyfit(np.log(h_values), np.log(L2_errors), 1)
coeff_Linf = np.polyfit(np.log(h_values), np.log(Linf_errors), 1)

# Extract slopes (exponents)
exp_L1 = coeff_L1[0]
exp_L2 = coeff_L2[0]
exp_Linf = coeff_Linf[0]

# Regression functions
fit_L1 = lambda x: np.exp(coeff_L1[1]) * x**exp_L1
fit_L2 = lambda x: np.exp(coeff_L2[1]) * x**exp_L2
fit_Linf = lambda x: np.exp(coeff_Linf[1]) * x**exp_Linf

# Plot errors and regression lines
plt.figure(figsize=(8, 6))
plt.scatter(h_values, L1_errors, color='b', marker='o', label="Erreur $L_1$")
plt.scatter(h_values, L2_errors, color='r', marker='o', label="Erreur $L_2$")
plt.scatter(h_values, Linf_errors, color='g', marker='o', label="Erreur $L_\infty$")

plt.loglog(h_values, fit_L1(h_values), 'b--', label=f"Fit: $L_1 \propto Δx^{{{exp_L1:.2f}}}$")
plt.loglog(h_values, fit_L2(h_values), 'r--', label=f"Fit: $L_2 \propto Δx^{{{exp_L2:.2f}}}$")
plt.loglog(h_values, fit_Linf(h_values), 'g--', label=f"Fit: $L_\infty \propto Δx^{{{exp_Linf:.2f}}}$")

# Position the equations separately on the plot
a = plt.text(h_values[1], L1_errors[1], f"$L_1 = {np.exp(coeff_L1[1]):.4f} \\times Δx^{{{exp_L1:.2f}}}$", fontsize=12, color='b')
b = plt.text(h_values[1], L2_errors[1], f"$L_2 = {np.exp(coeff_L2[1]):.4f} \\times Δx^{{{exp_L2:.2f}}}$", fontsize=12, color='r')
c = plt.text(h_values[1], Linf_errors[1], f"$L_\infty = {np.exp(coeff_Linf[1]):.4f} \\times Δx^{{{exp_Linf:.2f}}}$", fontsize=12, color='g')

# # Déplacer la zone de texte
a.set_position((0.05, 0.3))
b.set_position((0.05, 0.4))
c.set_position((0.03, 2))

# Rendre les axes plus gras
plt.gca().spines['bottom'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['right'].set_linewidth(2)
plt.gca().spines['top'].set_linewidth(2)

# Placer les marques de coche à l'intérieur et les rendre un peu plus longues
plt.tick_params(width=2, which='both', direction='in', top=True, right=True, length=6)


plt.title("Convergence d'ordre 1 des erreurs $L_1$, $L_2$ & $L_\infty$ en fonction de $\Delta r$", fontsize=14, fontweight='bold')
plt.xlabel("Taille de maille $\Delta r$ (m)", fontsize=12, fontweight='bold')
plt.ylabel("Erreur", fontsize=12, fontweight='bold')
plt.xscale('log')
plt.yscale('log')
plt.grid(True, which="both", linestyle="--")
plt.legend()
plt.show()

# Affichage des exposants obtenus
print(f"Exposant de convergence L1: {exp_L1:.4f}")
print(f"Exposant de convergence L2: {exp_L2:.4f}")
print(f"Exposant de convergence Linf: {exp_Linf:.4f}")



"--------------------QUESTION 1E)--------------------"

class parametres():
    D = 1
    Deff = 10**(-10)
    S = 2 * 10**-8
    Ce = 20
    Ri = 0
    Ro = D/2
    n = 5
    dr = (Ro - Ri)/(n-1)
    
prm = parametres()

Sol1 = FDM_1ordre(prm)

Sol2 = FDM_2ordre(prm)

r = np.linspace(prm.Ri, prm.Ro, 201)

f_anal = Sol_anal(r, prm)


plt.figure(3)
plt.scatter(Sol1[0], Sol1[1], color='b', marker='o', label="Numérique (FDM de 1er ordre)")
plt.scatter(Sol2[0], Sol2[1], color='r', marker='o', label="Numérique (FDM de 2e ordre)")
plt.plot(r, f_anal, 'k-', label="Analytique")
plt.xlabel("Position radiale r dans le pilier (m)")
plt.ylabel("Concentration C (mol/m³)")
plt.title("Profil de concentration stationnaire du \n processus de diffusion du sel dans le pilier de béton")
plt.legend()
plt.grid()
custom_yticks = [7.5, 10.625, 12, 14, 16, 18, 20] 
plt.yticks(custom_yticks)
plt.savefig('Ordre2.png', dpi=300)
plt.show()



"Vérification de l'erreur selon la FDM de 2e ordre selon 6 maillage différent"

nodes = [5, 10, 20, 50, 75, 100]  
L1_errors, L2_errors, Linf_errors = [], [], []
h_values = []

print(h_values)
print(L1_errors)

for n in nodes:
    prm.n = n
    prm.dr = (prm.Ro - prm.Ri) / (n-1)
    R_num, C_num = FDM_2ordre(prm)
    
    C_anal_num_nodes = Sol_anal(R_num, prm)   
    error_L1 = np.mean(np.abs(C_num - C_anal_num_nodes))
    error_L2 = np.sqrt(np.mean((C_num - C_anal_num_nodes)**2))
    error_Linf = np.max(np.abs(C_num - C_anal_num_nodes))

    L1_errors.append(error_L1)
    L2_errors.append(error_L2)
    Linf_errors.append(error_Linf)
    h_values.append(prm.dr)
        
L1_errors = np.array(L1_errors)
L2_errors = np.array(L2_errors)
Linf_errors = np.array(Linf_errors)
    

coeff_L1 = np.polyfit(np.log(h_values), np.log(L1_errors), 1)
coeff_L2 = np.polyfit(np.log(h_values), np.log(L2_errors), 1)
coeff_Linf = np.polyfit(np.log(h_values), np.log(Linf_errors), 1)

# Extract slopes (exponents)
exp_L1 = coeff_L1[0]
exp_L2 = coeff_L2[0]
exp_Linf = coeff_Linf[0]

# Regression functions
fit_L1 = lambda x: np.exp(coeff_L1[1]) * x**exp_L1
fit_L2 = lambda x: np.exp(coeff_L2[1]) * x**exp_L2
fit_Linf = lambda x: np.exp(coeff_Linf[1]) * x**exp_Linf

# Plot errors and regression lines
plt.figure(figsize=(8, 6))
plt.scatter(h_values, L1_errors, color='b', marker='o', label="Erreur $L_1$")
plt.scatter(h_values, L2_errors, color='r', marker='o', label="Erreur $L_2$")
plt.scatter(h_values, Linf_errors, color='g', marker='o', label="Erreur $L_\infty$")

plt.loglog(h_values, fit_L1(h_values), 'b--', label=f"Fit: $L_1 \propto Δx^{{{exp_L1:.2f}}}$")
plt.loglog(h_values, fit_L2(h_values), 'r--', label=f"Fit: $L_2 \propto Δx^{{{exp_L2:.2f}}}$")
plt.loglog(h_values, fit_Linf(h_values), 'g--', label=f"Fit: $L_\infty \propto Δx^{{{exp_Linf:.2f}}}$")

# Position the equations separately on the plot
a = plt.text(h_values[1], L1_errors[1], f"$L_1 = {np.exp(coeff_L1[1]):.4f} \\times Δx^{{{exp_L1:.2f}}}$", fontsize=12, color='b')
b = plt.text(h_values[1], L2_errors[1], f"$L_2 = {np.exp(coeff_L2[1]):.4f} \\times Δx^{{{exp_L2:.2f}}}$", fontsize=12, color='r')
c = plt.text(h_values[1], Linf_errors[1], f"$L_\infty = {np.exp(coeff_Linf[1]):.4f} \\times Δx^{{{exp_Linf:.2f}}}$", fontsize=12, color='g')

# Déplacer la zone de texte
a.set_position((0.02, 5e-14))
b.set_position((0.05, 13e-14))
c.set_position((0.04, 5e-13))

# Rendre les axes plus gras
plt.gca().spines['bottom'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['right'].set_linewidth(2)
plt.gca().spines['top'].set_linewidth(2)

# Placer les marques de coche à l'intérieur et les rendre un peu plus longues
plt.tick_params(width=2, which='both', direction='in', top=True, right=True, length=6)


plt.title("Convergence d'ordre 2 des erreurs $L_1$, $L_2$ & $L_\infty$ en fonction de $\Delta r$", fontsize=14, fontweight='bold')
plt.xlabel("Taille de maille $\Delta r$ (m)", fontsize=12, fontweight='bold')
plt.ylabel("Erreur", fontsize=12, fontweight='bold')
plt.xscale('log')
plt.yscale('log')
plt.grid(True, which="both", linestyle="--")
plt.legend()
plt.show()

# Affichage des exposants obtenus
print(f"Exposant de convergence L1: {exp_L1:.4f}")
print(f"Exposant de convergence L2: {exp_L2:.4f}")
print(f"Exposant de convergence Linf: {exp_Linf:.4f}")





















