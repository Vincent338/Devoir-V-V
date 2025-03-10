import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

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
plt.title("Profils de concentration à plusieurs temps")
plt.grid()
plt.legend()
plt.savefig("profils_concentration")
plt.show()






##### MMS #####
def MMS():
    # Définition des variables symboliques
    t, r = sp.symbols('t r')
    # Définir les paramètres symboliques
    Ce_sym, C0_sym, Ro_sym, k_sym = sp.symbols('Ce C0 Ro k')
    
    # solution manufacturée
    C_MMS = Ce_sym + (C0_sym - Ce_sym) * sp.exp(t*k_sym* 1e-4) * (1 - r**2/Ro_sym**2)
    
    # Substituer les valeurs numériques dans l'expression
    params = {Ce_sym: 20,   # Ce = 20 mol/m³
              C0_sym: 1,    # C0 fixé à 0 
              Ro_sym: 0.5,  # Ro = 0.5 m
              k_sym: 4e-9}  # k = 4e-9
              
    C_MMS_num = C_MMS.subs(params)
    
    
    # Calcul des dérivées
    C_t = sp.diff(C_MMS, t)
    C_r = sp.diff(C_MMS, r)
    C_rr = sp.diff(sp.diff(C_MMS, r), r)
    
    # Définition du terme source S(t, r)
    Deff_val = 1e-10
    source = C_t - Deff_val * (C_rr + (1/r) * C_r) + k_sym * C_MMS
    source_num = source.subs(params)
    
    # Conditions aux limites et initiales
    C_initial = C_MMS.subs(t, 0)
    C_boundary_Ro = C_MMS.subs(r, Ro)
    dCdr_boundary_0 = C_r.subs(r, 0)
    
    # Affichage des équations
    print("Dérivée en temps :")
    print(C_t)
    print("Dérivée première en r :")
    print(C_r)
    print("Dérivée seconde en r :")
    print(C_rr)
    print("Terme source :")
    print(source)
    print("\nCondition initiale C(0, r) :")
    print(C_initial)
    print("\nCondition frontière Dirichlet C(t, Ro) :")
    print(C_boundary_Ro)
    print("\nCondition frontière Neumann dC/dr(t, 0) :")
    print(dCdr_boundary_0)
    
    # Conversion en fonctions Python
    f_C_MMS = sp.lambdify([t, r], C_MMS_num, "numpy")
    f_source = sp.lambdify([t, r], source_num, "numpy")
    
    return(f_C_MMS, f_source, source)

f_C_MMS, f_source, source = MMS()

##### Solution instationnaire euler implicite MMS #####

def solinsta_mms(Deff, k, Ce, Ro, n, dt, t_final, f_source):    
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
    B[-1] = Ce
    
        
    t_array=[0]
    t=0
    for n_t in range(1, nt):
        # Pour les points intérieurs : on utilise la solution de l'instant précédent
        for i in range(1, n-1):
            B[i] = - C[n_t - 1, i] - f_source(t, r[i]) * dt
        # Pour le bord en r=0, on impose dC/dr = 0
        B[0] = 0
        # Pour le bord en r=Ro, on impose la valeur Dirichlet Ce
        B[-1] = Ce
        
        # Résolution du système A * C_new = B
        C[n_t, :] = np.linalg.solve(A, B)
        t+=dt
        t_array.append(t)
    return(C,t_array)

C_solinsta_mms, t_array= solinsta_mms(Deff, k, Ce, Ro, n, dt, t_final, f_source)

alpha=f_C_MMS(t_final,r)
# Affichage des résultats
plt.figure(figsize=(8, 6), dpi=150)
for t in (t_array[0],t_array[nt//2] ,t_array[-1]):
    plt.plot(r, f_C_MMS(t,r), label=f"C_MMS, {t}")
    plt.plot(r, C_solinsta_mms[i,:],".", label=f"solution numérique, {t}")
plt.title(f"Profil de concentration à {t} s")
plt.xlabel("Rayon [m]")
plt.ylabel("Concentration [mol/m³]")
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.show()

# Conversion du terme source en code C et Fortran
terme_source_c = sp.ccode(source)
terme_source_fortran = sp.fcode(source)

# Affichage des résultats
print("Code C :")
print(terme_source_c)
print("\nCode Fortran :")
print(terme_source_fortran)

##### Analyse de convergence (incomplet) #####

def compute_error_norms(C_mms, C_exact, n):
    error = np.abs(C_mms - C_exact)
    L1 = np.sum(error) / n
    L2 = np.sqrt(np.sum(error**2) / n)
    Linf = np.max(error)
    return L1, L2, Linf


def convergence_espace(Deff, k, Ce, Ro, dt, t_final, f_source, f_C_MMS):
    # n_values = np.arange(10,1010,100)
    n_values =[10,100]
    errors_L1 = []
    errors_L2 = []
    errors_Linf = []
    dr_values = []
    
    
    for n in n_values:
        r = np.linspace(0, Ro, n)
        
        dr = Ro / (n - 1)
        dr_values.append(dr)
        
        # Calcul de la solution analytique
        C_mms = f_C_MMS(t_final, r)
        
        # Calcul de la solution numérique
        C_exact = solinsta_mms(Deff, k, Ce, Ro, n, dt, t_final, f_source)
        
        L1, L2, Linf = compute_error_norms(C_exact, C_mms, n)
        
        print(n, "MMS: ", C_mms, "\n Sol insta: ", C_exact)
        plt.plot(r, C_mms,'.', label="C_mms")
        plt.plot(r, C_exact,'.', label="Sol insta")
        plt.legend()
        plt.show()
        errors_L1.append(L1)
        errors_L2.append(L2)
        errors_Linf.append(Linf)
    
    
    coefficientsL1 = np.polyfit(np.log(dr_values), np.log(errors_L1), 1)
    coefficientsL2 = np.polyfit(np.log(dr_values), np.log(errors_L2), 1)
    coefficientsLinf = np.polyfit(np.log(dr_values), np.log(errors_Linf), 1)

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
    extrapolated_valueL1 = fit_functionL1(dr_values[-1])
    extrapolated_valueL2 = fit_functionL2(dr_values[-1])
    extrapolated_valueLinf = fit_functionLinf(dr_values[-1])

    # Tracé des erreurs en fonction de la taille de la maille
    plt.figure(figsize=(8, 6), dpi=250)
    plt.scatter(dr_values, errors_L1, marker='o', color='r', label=r'Erreur $L_1$')
    plt.scatter(dr_values, errors_L2, marker='o', color='b', label=r'Erreur $L_2$')
    plt.scatter(dr_values, errors_Linf, marker='o', color='g', label=r'Erreur $L_∞$')

    plt.plot(dr_values, fit_functionL1(dr_values), linestyle='--', color='r', label=f'Régression $L_1$ en loi de puissance (pente = {exponentL1:.2f})')
    plt.plot(dr_values, fit_functionL2(dr_values), linestyle='--', color='b', label=f'Régression $L_2$ en loi de puissance (pente = {exponentL2:.2f})')
    plt.plot(dr_values, fit_functionLinf(dr_values), linestyle='--', color='g', label=f'Régression $L_∞$ en loi de puissance (pente = {exponentLinf:.2f})')


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

    plt.title(r"Convergences d'espace des erreurs $L_1$ , $L_2$ et $L_∞$ en fonction de $Δr$")
    plt.xlabel("Taille de maille $Δr$ (m)")
    plt.ylabel("Erreur")
    plt.grid(which="both", linestyle="--", linewidth=0.5)
    plt.legend()

    plt.tight_layout()
    plt.show()


def convergence_temps(Deff, k, Ce, Ro, n, t_final):
    dt_values = [1e4, 5e3, 2.5e3, 1.25e3]  # par exemple
    errors_L1 = []
    errors_L2 = []
    errors_Linf = []
    
    # Pour réduire l'erreur spatiale, on choisit n fixe grand
    dr = Ro / (n - 1)
    
    # Définir la solution manufacturée exacte à t_final
    t_sym, r_sym = sp.symbols('t r')
    C_MMS_expr = MMS(Ce, 0, Ro, k) [0]
    f_C_MMS = sp.lambdify([t_sym, r_sym], C_MMS_expr, "numpy")
    
    r_num = np.linspace(0, Ro, n)
    
    for dt in dt_values:
        # Calcul de la solution numérique
        r_num, C_num, t_sim = solinsta(Deff, k, Ce, Ro, n, dt, t_final)
        # Calcul de la solution exacte
        C_exact = f_C_MMS(t_final, r_num)
        L1, L2, Linf = compute_error_norms(C_num, C_exact, dr)
        errors_L1.append(L1)
        errors_L2.append(L2)
        errors_Linf.append(Linf)
        print(f"dt = {dt:.4e} : L1 = {L1:.4e}, L2 = {L2:.4e}, Linf = {Linf:.4e}")
    
    # Tracer les erreurs en fonction de dt
    plt.figure(figsize=(8,6))
    plt.loglog(dt_values, errors_L1, '-o', label='L1')
    plt.loglog(dt_values, errors_L2, '-o', label='L2')
    plt.loglog(dt_values, errors_Linf, '-o', label='Linf')
    plt.xlabel("Pas de temps dt")
    plt.ylabel("Erreur")
    plt.title("Analyse de convergence en temps")
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.show()
    

# # Lancer une analyse de convergence en espace
convergence_espace(Deff, k, Ce, Ro, 40000, t_final, f_source, f_C_MMS)

# Pour l'analyse en temps, choisissez n suffisamment grand
n_refined = 80
convergence_temps(Deff, k, Ce, Ro, n_refined, t_final)