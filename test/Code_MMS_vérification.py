import os
import matplotlib.pyplot as plt
import numpy as np

def FDM_with_MMS(prm):
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
    
    C_stock = []
    C_stock_MMS = []
    
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
            
            B[i] = (Ce*np.exp(4e9*k) - Ro**2 + R[i]**2)*np.exp(-4e9*k) + dt*(-Ce*k - 4*Deff*np.exp(-k*(4e9 - t0)) + 2*k*(Ce*np.exp(k*(4e9 - t0)) - Ro**2 + R[i]**2)*np.exp(-k*(4e9 - t0)))
                
    #print(B)
    #print(M)
    
    while (t0 <= tf) == True:
        
        "Résolution du système matriciel"        
        C = np.linalg.solve(M, B)   
        
        C_stock.append(C)
        C_stock_MMS.append(MMS(t0, prm))
        
        t0 = t0 + dt
        
        #print(C)
                    
        B = C.copy()
        
        B += dt*(-Ce*k - 4*Deff*np.exp(-k*(4e9 - t0)) + 2*k*(Ce*np.exp(k*(4e9 - t0)) - Ro**2 + R**2)*np.exp(-k*(4e9 - t0)))
            
        B[0] = 0
        B[-1] = Ce
        #print(B)
        t.append(t0)
                
    return C, R, t, np.array(C_stock), np.array(C_stock_MMS)

def MMS(t, prm):
    
    "Calcul de la solution analytique MMS selon la position radiale sur le domaine r"
    
    Ce = prm.Ce
    Ro = prm.Ro
    k = prm.k
    r = np.linspace(0, prm.Ro, prm.n)
    
    Sol = np.exp(-k*(4e9-t))*(r**2 + np.exp(k*(4e9-t))*Ce - Ro**2)
    
    return Sol

# def source():
    

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
        self.dt = 3.1536e7
        self.tf = 4e9
    
prm = Parametres()

Da = (prm.k*prm.Ro**2 + prm.k*prm.Ce - 4*prm.Deff)/prm.Deff

"--------------------QUESTION 1D)--------------------"

C, R, t, C_stock, _ = FDM_with_MMS(prm)

r = np.linspace(0, prm.Ro, prm.n)
#ta = np.linspace(0, prm.tf, 201)
f_anal = MMS(4e9, prm)

plt.figure(1)
plt.scatter(R, C, color='b', marker='o', label="Numérique (FDM de 1er ordre)")
plt.plot(r, f_anal, 'k-', label="Analytique")
plt.xlabel("Position radiale r dans le pilier (m)")
plt.ylabel("Concentration C (mol/m³)")
plt.title("Profil de concentration stationnaire du \n processus de diffusion du sel dans le pilier de béton")
plt.legend()
plt.grid()
#custom_yticks = [7.5, 10.625, 12, 14, 16, 18, 20]  # Change these values as needed
#plt.yticks(custom_yticks)
#plt.savefig('Ordre1.png', dpi=300)
plt.show()
    
time = [0, 800e6, 2*800e6, 3*800e6, 4*800e6, 5*800e6]

plt.figure(2)
for t in time:
    a = MMS(t, prm)
    plt.plot(r, a, label=f"t = {t}")
    
plt.xlabel("Position radiale r dans le pilier (m)")
plt.ylabel("Concentration C (mol/m³)")
plt.title("Profil de concentration de la solution manufacturée \n à différents temps $t$")
plt.grid()
plt.legend()
plt.savefig('MMS.png', dpi=300)
plt.show()


plt.figure(3)
for t in time:
    a = (-prm.Ce*prm.k - 4*prm.Deff*np.exp(-prm.k*(4e9 - t)) + 2*prm.k*(prm.Ce*np.exp(prm.k*(4e9 - t)) - prm.Ro**2 + R**2)*np.exp(-prm.k*(4e9 - t)))
    plt.plot(R, a, label=f"t = {t}")
plt.xlabel("Position radiale r dans le pilier (m)")
plt.ylabel("Concentration C (mol/m³)")
plt.title("Profil de concentration du terme source \n à différents temps $t$")
plt.grid()
plt.legend()
plt.savefig('source.png', dpi=300)
plt.show()



def compute_error_norms(C_mms, C_exact, n):
    error = np.abs(C_mms - C_exact)
    L1 = np.sum(error) / n
    #print(L1)
    L2 = np.sqrt(np.sum(error**2) / n)
    Linf = np.max(error)
    return L1, L2, Linf


def convergence_espace(Deff, k, Ce, Ro, dt, t_final, MMS):
    n_values = np.linspace(100, 1000, 10, dtype = int)
    errors_L1 = []
    errors_L2 = []
    errors_Linf = []
    dr_values = []
    
    
    for n in n_values:
        prm.n = n
        r = np.linspace(0, prm.Ro, prm.n)

        dr = Ro / (n - 1)
        dr_values.append(dr)
        
        # Calcul de la solution analytique
        C_mms = MMS(t_final, prm)
        
        # Calcul de la solution numérique
        C_exact,_,_, _, _ = FDM_with_MMS(prm)

        
        L1, L2, Linf = compute_error_norms(C_exact, C_mms, n)
        
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
    plt.scatter(dr_values, errors_L1, marker='.', color='r', label=r'Erreur $L_1$')
    plt.scatter(dr_values, errors_L2, marker='.', color='b', label=r'Erreur $L_2$')
    plt.scatter(dr_values, errors_Linf, marker='.', color='g', label=r'Erreur $L_∞$')

    plt.plot(dr_values, fit_functionL1(dr_values), linestyle='--', color='r', label=f'Régression $L_1$ en loi de puissance (pente = {exponentL1:.2f})')
    plt.plot(dr_values, fit_functionL2(dr_values), linestyle='--', color='b', label=f'Régression $L_2$ en loi de puissance (pente = {exponentL2:.2f})')
    plt.plot(dr_values, fit_functionLinf(dr_values), linestyle='--', color='g', label=f'Régression $L_∞$ en loi de puissance (pente = {exponentLinf:.2f})')


    equation_textL1 = f'$L_1 = {np.exp(coefficientsL1[1]):.2e} \\times Δx^{{{exponentL1:.2f}}}$'
    equation_text_objL1 = plt.text(0.05, 0.05, equation_textL1, fontsize=12, transform=plt.gca().transAxes, color='r')
    equation_text_objL1.set_position((0.25, 0.15))

    equation_textL2 = f'$L_2 = {np.exp(coefficientsL2[1]):.2e} \\times Δx^{{{exponentL2:.2f}}}$'
    equation_text_objL2 = plt.text(0.05, 0.05, equation_textL2, fontsize=12, transform=plt.gca().transAxes, color='b')
    equation_text_objL2.set_position((0.20, 0.37))

    equation_textLinf = f'$L_∞ = {np.exp(coefficientsLinf[1]):.2e} \\times Δx^{{{exponentLinf:.2f}}}$'
    equation_text_objLinf = plt.text(0.05, 0.05, equation_textLinf, fontsize=12, transform=plt.gca().transAxes, color='g')
    equation_text_objLinf.set_position((0.25, 0.60))

    plt.xscale('log')
    plt.yscale('log')

    plt.title(r"Ordre de convergence en espace des erreurs $L_1$ , $L_2$ et $L_∞$ en fonction de $Δr$")
    plt.xlabel("Taille de maille $Δr$ (m)")
    plt.ylabel("Erreur")
    plt.grid(which="both", linestyle="--", linewidth=0.5)
    plt.legend()

    plt.tight_layout()
    plt.savefig('Ordre2.png', dpi=300)
    plt.show()

convergence_espace(prm.Deff, prm.k, prm.Ce, prm.Ro, 400000, prm.tf, MMS)

def convergence_temps(Deff, k, Ce, Ro, dt, t_final, MMS):
    dt_values = np.linspace(400000, 40000000, 5)
    errors_L1 = []
    errors_L2 = []
    errors_Linf = []
    dr_values = []
    
    prm.n = 1000
    for t in dt_values:
        prm.dt = t
        print(t)
        r = np.linspace(0, prm.Ro, prm.n)

        dr_values.append(t)
        
        # Calcul de la solution analytique
        C_mms = MMS(t_final, prm)
        
        # Calcul de la solution numérique
        C_exact,_,_,_, _ = FDM_with_MMS(prm)
 
        
        L1, L2, Linf = compute_error_norms(C_exact, C_mms, int(prm.tf/t))
        
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
    plt.scatter(dr_values, errors_L1, marker='.', color='r', label=r'Erreur $L_1$')
    plt.scatter(dr_values, errors_L2, marker='.', color='b', label=r'Erreur $L_2$')
    plt.scatter(dr_values, errors_Linf, marker='.', color='g', label=r'Erreur $L_∞$')

    plt.plot(dr_values, fit_functionL1(dr_values), linestyle='--', color='r', label=f'Régression $L_1$ en loi de puissance (pente = {exponentL1:.2f})')
    plt.plot(dr_values, fit_functionL2(dr_values), linestyle='--', color='b', label=f'Régression $L_2$ en loi de puissance (pente = {exponentL2:.2f})')
    plt.plot(dr_values, fit_functionLinf(dr_values), linestyle='--', color='g', label=f'Régression $L_∞$ en loi de puissance (pente = {exponentLinf:.2f})')


    equation_textL1 = f'$L_1 = {np.exp(coefficientsL1[1]):.2e} \\times Δx^{{{exponentL1:.2f}}}$'
    equation_text_objL1 = plt.text(0.05, 0.05, equation_textL1, fontsize=12, transform=plt.gca().transAxes, color='r')
    equation_text_objL1.set_position((0.25, 0.15))

    equation_textL2 = f'$L_2 = {np.exp(coefficientsL2[1]):.2e} \\times Δx^{{{exponentL2:.2f}}}$'
    equation_text_objL2 = plt.text(0.05, 0.05, equation_textL2, fontsize=12, transform=plt.gca().transAxes, color='b')
    equation_text_objL2.set_position((0.20, 0.37))

    equation_textLinf = f'$L_∞ = {np.exp(coefficientsLinf[1]):.2e} \\times Δx^{{{exponentLinf:.2f}}}$'
    equation_text_objLinf = plt.text(0.05, 0.05, equation_textLinf, fontsize=12, transform=plt.gca().transAxes, color='g')
    equation_text_objLinf.set_position((0.25, 0.60))

    plt.xscale('log')
    plt.yscale('log')

    plt.title(r"Ordre de convergence en temps des erreurs $L_1$ , $L_2$ et $L_∞$ en fonction de $Δt$")
    plt.xlabel("Pas de temps $Δt$ (s)")
    plt.ylabel("Erreur")
    plt.grid(which="both", linestyle="--", linewidth=0.5)
    plt.legend()

    plt.tight_layout()
    plt.savefig('Ordre1.png', dpi=300)
    plt.show()

convergence_temps(prm.Deff, prm.k, prm.Ce, prm.Ro, 400000, prm.tf, MMS)
