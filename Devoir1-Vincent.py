# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 22:06:09 2025

@author: Vincent
"""
import numpy as np
import matplotlib.pyplot as plt

# N = 20 #Nombre de noeuds

# # Plage de valeurs pour N
N = np.arange(5,35,5)

# Initialisation des listes d'erreurs
L1_errors = []
L2_errors = []
Linf_errors = []

L1_errors2 = []
L2_errors2 = []
Linf_errors2 = []

deltar_val1 = []
deltar_val2 = []

# Paramètres physiques du problème
R = 0.5 #Rayon du pilier
Deff = 10**(-10) # Coefficient de diffusion effectif
S = 2*(10**(-8)) #Quantité de sel
Ce = 20 # Concentration de l'eau

####### D) Ordre 1 #########

# Boucle sur différentes valeurs de N
for i in range(len(N)):
    n=N[i]
    r=np.linspace(0, R, n)
    deltar=r[1]-r[0] # Taille de la maille
    deltar_val1.append(deltar)
    
    # Initialisation des matrices et vecteurs
    A=np.zeros((n,n))
    b=np.zeros(n)
    C=np.zeros(n)
    
    # Conditions aux limites
    A[0,0]=1
    A[0,1]=-1
    
    # Construction de la matrice A et du vecteur b
    for i in range(1,n-1):
        A[i,i-1]= 1/deltar**2
        A[i,i]=-2/deltar**2 - 1/(r[i]*deltar)
        A[i,i+1]=1/deltar**2 + 1/(r[i]*deltar)
        
        b[i]= S/Deff
        
    # Condition à la frontière extérieure
    A[n-1,n-1]=1
    b[n-1]=Ce  
    
    # Résolution du système
    C = np.linalg.solve(A, b)
    
    
    # Solution analytique
    C_analytique = (S / (4 * Deff) * R**2) * ((r**2 / R**2) - 1) + Ce
    
    # Calcul des erreurs
    erreur_abs = np.abs(C - C_analytique)
    L1 = np.sum(erreur_abs) / n
    L2 = np.sqrt(np.sum(erreur_abs**2) / n)
    Linf = np.max(erreur_abs)
    
    # Stockage des erreurs
    L1_errors.append(L1)
    L2_errors.append(L2)
    Linf_errors.append(Linf)
    

# Conversion des listes en tableaux numpy pour faciliter les calculs
deltar_val = np.array(deltar_val1)
L1_errors = np.array(L1_errors)
L2_errors = np.array(L2_errors)
Linf_errors = np.array(Linf_errors)


coefficientsL1 = np.polyfit(np.log(deltar_val1), np.log(L1_errors), 1)
coefficientsL2 = np.polyfit(np.log(deltar_val1), np.log(L2_errors), 1)
coefficientsLinf = np.polyfit(np.log(deltar_val1), np.log(Linf_errors), 1)

exponentL1 = coefficientsL1[0]
exponentL2 = coefficientsL2[0]
exponentLinf = coefficientsLinf[0]

# Fonctions d'ajustement pour le tracé des erreurs
fit_function_logL1 = lambda x: exponentL1 * x + coefficientsL1[1]
fit_function_logL2 = lambda x: exponentL2 * x + coefficientsL2[1]
fit_function_logLinf = lambda x: exponentLinf * x + coefficientsLinf[1]

fit_functionL1 = lambda x: np.exp(fit_function_logL1(np.log(x)))
fit_functionL2 = lambda x: np.exp(fit_function_logL2(np.log(x)))
fit_functionLinf = lambda x: np.exp(fit_function_logLinf(np.log(x)))

# Extrapoler la valeur prédite pour la dernière valeur de h_values
extrapolated_value = fit_functionL1(deltar_val1[-1])
extrapolated_value = fit_functionL1(deltar_val1[-1])
extrapolated_value = fit_functionL1(deltar_val1[-1])

# Tracé des erreurs en fonction de la taille de la maille
plt.figure(figsize=(8, 6), dpi=250)
plt.scatter(deltar_val1, L1_errors, marker='o', color='r', label=r'Erreur $L_1$')
plt.scatter(deltar_val1, L2_errors, marker='o', color='b', label=r'Erreur $L_2$')
plt.scatter(deltar_val1, Linf_errors, marker='o', color='g', label=r'Erreur $L_∞$')

plt.plot(deltar_val1, fit_functionL1(deltar_val1), linestyle='--', color='r', label=f'Régression $L_1$ en loi de puissance (pente = {exponentL1:.2f})')
plt.plot(deltar_val1, fit_functionL2(deltar_val1), linestyle='--', color='b', label=f'Régression $L_2$ en loi de puissance (pente = {exponentL2:.2f})')
plt.plot(deltar_val1, fit_functionLinf(deltar_val1), linestyle='--', color='g', label=f'Régression $L_∞$ en loi de puissance (pente = {exponentLinf:.2f})')


equation_textL1 = f'$L_1 = {np.exp(coefficientsL1[1]):.2f} \\times Δx^{{{exponentL1:.2f}}}$'
equation_text_objL1 = plt.text(0.05, 0.05, equation_textL1, fontsize=12, transform=plt.gca().transAxes, color='r')
equation_text_objL1.set_position((0.25, 0.15))

equation_textL2 = f'$L_2 = {np.exp(coefficientsL2[1]):.2f} \\times Δx^{{{exponentL2:.2f}}}$'
equation_text_objL2 = plt.text(0.05, 0.05, equation_textL2, fontsize=12, transform=plt.gca().transAxes, color='b')
equation_text_objL2.set_position((0.20, 0.37))

equation_textLinf = f'$L_∞ = {np.exp(coefficientsLinf[1]):.2f} \\times Δx^{{{exponentLinf:.2f}}}$'
equation_text_objLinf = plt.text(0.05, 0.05, equation_textLinf, fontsize=12, transform=plt.gca().transAxes, color='g')
equation_text_objLinf.set_position((0.25, 0.60))

plt.xscale('log')
plt.yscale('log')

plt.title(r"Convergences d'ordre 1 des erreurs $L_1$ , $L_2$ et $L_∞$ en fonction de $Δr$")
plt.xlabel("Taille de maille $Δr$ (m)")
plt.ylabel("Erreur")
plt.grid(which="both", linestyle="--", linewidth=0.5)
plt.legend()

plt.tight_layout()
plt.show()


####### E) Ordre 2 #########
for i in range(len(N)):
    n=N[i]
    r=np.linspace(0, R, n)
    deltar=r[1]-r[0] # Taille de la maille
    deltar_val2.append(deltar)

    # Initialisation des matrices et vecteurs
    A2=np.zeros((n,n))
    b=np.zeros(n)
    C2=np.zeros(n)
    
    # Conditions aux limites
    A2[0, 0] = -3
    A2[0, 1] = 4
    A2[0, 2] = -1
    
    # Construction de la matrice A et du vecteur b
    for i in range(1,n-1):
        A2[i,i-1]= 1/deltar**2 - 1/(2*r[i]*deltar)
        A2[i,i]=-2/deltar**2
        A2[i,i+1]= 1/deltar**2 + 1/(2*r[i]*deltar)
        
        b[i]=S/Deff
    
    # Condition à la frontière extérieure
    A2[n-1,n-1]=1
    b[n-1] = Ce
    
    # Résolution du système
    C2 = np.linalg.solve(A2, b)
    
    # Solution analytique
    C_analytique = (S / (4 * Deff) * R**2) * ((r**2 / R**2) - 1) + Ce
    
    #Calcul des erreurs
    erreur_abs = np.abs(C2 - C_analytique)
    L1_2 = np.sum(erreur_abs) / n
    L2_2 = np.sqrt(np.sum(erreur_abs**2) / n)
    Linf_2 = np.max(erreur_abs)
    
    # Stockage des erreurs
    L1_errors2.append(L1_2)
    L2_errors2.append(L2_2)
    Linf_errors2.append(Linf_2)
    
#Graphique de distribution de concentration pour les deux méthodes
plt.figure(figsize=(8, 6), dpi=150)
plt.plot(r, C, 'g.', linewidth=2, markersize=8, label="Solution Numérique Ordre 1")
plt.plot(r, C2, 'b.', linewidth=2, markersize=8, label="Solution Numérique Ordre 2")
plt.plot(r, C_analytique, 'r--', linewidth=2, label="Solution Analytique")
plt.xlabel("Rayon (m)", fontsize=14)
plt.ylabel("Concentration (mol/m³)", fontsize=14)
plt.title("Concentration de sel dans la structure poreuse d’un pilier de béton", fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

# Conversion des listes en tableaux numpy pour faciliter les calculs
deltar_val2 = np.array(deltar_val2)
L1_errors2 = np.array(L1_errors2)
L2_errors2 = np.array(L2_errors2)
Linf_errors2 = np.array(Linf_errors2)

# Ajustement logarithmique pour estimer l'ordre de convergence
coefficientsL1_2 = np.polyfit(np.log(deltar_val2), np.log(L1_errors2), 1)
coefficientsL2_2 = np.polyfit(np.log(deltar_val2), np.log(L2_errors2), 1)
coefficientsLinf_2 = np.polyfit(np.log(deltar_val2), np.log(Linf_errors2), 1)

exponentL1_2 = coefficientsL1_2[0]
exponentL2_2 = coefficientsL2_2[0]
exponentLinf_2 = coefficientsLinf_2[0]

# Fonctions d'ajustement pour le tracé des erreurs
fit_function_logL1_2 = lambda x: exponentL1_2 * x + coefficientsL1_2[1]
fit_function_logL2_2 = lambda x: exponentL2_2 * x + coefficientsL2_2[1]
fit_function_logLinf_2 = lambda x: exponentLinf_2 * x + coefficientsLinf_2[1]

fit_functionL1_2 = lambda x: np.exp(fit_function_logL1_2(np.log(x)))
fit_functionL2_2 = lambda x: np.exp(fit_function_logL2_2(np.log(x)))
fit_functionLinf_2 = lambda x: np.exp(fit_function_logLinf_2(np.log(x)))

# Extrapoler la valeur prédite pour la dernière valeur de h_values
extrapolated_valueL1_2 = fit_functionL1_2(deltar_val2[-1])
extrapolated_valueL2_2 = fit_functionL2_2(deltar_val2[-1])
extrapolated_valueLinf_2 = fit_functionLinf_2(deltar_val2[-1])

# Tracé des erreurs en fonction de la taille de la maille
plt.figure(figsize=(8, 6), dpi=250)
plt.scatter(deltar_val2, L1_errors2, marker='o', color='r', label='Erreur L1')
plt.scatter(deltar_val2, L2_errors2, marker='o', color='b', label='Erreur L2')
plt.scatter(deltar_val2, Linf_errors2, marker='o', color='g', label='Erreur Linf')

plt.plot(deltar_val2, fit_functionL1_2(deltar_val2), linestyle='--', color='r', label=f'Régression $L_1$ en loi de puissance (pente = {exponentL1_2:.2f})')
plt.plot(deltar_val2, fit_functionL2_2(deltar_val2), linestyle='--', color='b', label=f'Régression $L_2$ en loi de puissance (pente = {exponentL2_2:.2f})')
plt.plot(deltar_val2, fit_functionLinf_2(deltar_val2), linestyle='--', color='g', label=f'Régression $L_∞$ en loi de puissance (pente = {exponentLinf_2:.2f})')


equation_textL1_2 = f'$L_1 = {np.exp(coefficientsL1[1]):.2f} \\times Δx^{{{exponentL1:.2f}}}$'
equation_text_objL1_2 = plt.text(0.05, 0.05, equation_textL1_2, fontsize=12, transform=plt.gca().transAxes, color='r')
equation_text_objL1_2.set_position((0.25, 0.25))

equation_textL2_2 = f'$L_2 = {np.exp(coefficientsL2[1]):.2f} \\times Δx^{{{exponentL2:.2f}}}$'
equation_text_objL2_2 = plt.text(0.05, 0.05, equation_textL2_2, fontsize=12, transform=plt.gca().transAxes, color='b')
equation_text_objL2_2.set_position((0.18, 0.53))

equation_textLinf_2 = f'$L_∞ = {np.exp(coefficientsLinf[1]):.2f} \\times Δx^{{{exponentLinf:.2f}}}$'
equation_text_objLinf_2 = plt.text(0.05, 0.05, equation_textLinf_2, fontsize=12, transform=plt.gca().transAxes, color='g')
equation_text_objLinf_2.set_position((0.2, 0.74))

plt.xscale('log')
plt.yscale('log')


plt.xlabel("Taille de maille $Δr$ (m)")
plt.ylabel("Erreur")
plt.title(r"Convergences d'ordre 2 des erreurs $L_1$ , $L_2$ et $L_∞$ en fonction de $Δr$")
plt.grid(which="both", linestyle="--", linewidth=0.5)
plt.legend()

plt.tight_layout()
plt.show()

