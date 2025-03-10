import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

t, r = sp.symbols('t r')
Deff, k, Ce, R = sp.symbols('Deff k Ce R')

"solution MMS"
C_MMS = sp.exp(-k*(4e9-t))*(r**2 + sp.exp(k*(4e9-t))*Ce - R**2)

"Calcul des dérivées"
C_t = sp.diff(C_MMS, t)
C_r = sp.diff(C_MMS, r)
C_rr = sp.diff(sp.diff(C_MMS, r), r)

"Calcul du terme source S(t, r)"
source = C_t - Deff*(C_rr + 1/r * C_r) + k*C_MMS

"Conversion en fonctions Python"
f_C_MMS = sp.lambdify([t, r, k, Ce, R], C_MMS, "numpy")
f_source = sp.lambdify([t, r, Deff, k, Ce, R], source, "numpy")

"Condition initiale et limites"
C_initial = C_MMS.subs(t, 0)
C_in = C_r.subs(r, 0)
C_out = C_MMS.subs(r, R)

# Affichage des dérivées
print("Dérivée en temps :")
print(C_t)
print("Dérivée première :")
print(C_r)
print("Dérivée seconde :")
print(C_rr)
print("Terme source :")
print(source)
print("\nCondition initiale C(t = 0, r) :")
print(C_initial)
print("\nCondition frontière Neumann dC/dr(t, r = 0) :")
print(C_in)
print("\nCondition frontière Dirichlet C(t,r = R) :")
print(C_out)

#Visualiser les fonctions sur un maillage  
# taille du domaine
tmin, tmax = 0, 5
rmin, rmax = 0, 0.5
nt, nr = 100, 100

# Établir une grille régulière de points d'interpolation    
tdom = np.linspace(tmin,tmax, nt)
rdom = np.linspace(rmin,rmax, nr)
ti, ri = np.meshgrid(tdom, rdom)

# Évaluer la fonction MMS et le terme source sur le maillage
z_MMS    = f_C_MMS(ti, ri, 4e-9, 20, 0.5)
z_source = f_source(ti, ri, 1e-8, 4e-9, 20, 0.5)

# Tracer les résultats
plt.contourf(ri, ti, z_MMS, levels=50)
plt.colorbar()
plt.title('Fonction MMS')
plt.xlabel('r')
plt.ylabel('t')
plt.show()

plt.contourf(ri, ti, z_source, levels=50)
plt.colorbar()
plt.title('Terme source')
plt.xlabel('r')
plt.ylabel('t')
plt.show()

# Convertir le terme source en code C
terme_source_c = sp.ccode(source)

# Convertir le terme source en code Fortran
terme_source_fortran = sp.fcode(source)

# Convertir l'expression en une chaîne LaTeX
equation_latex = sp.latex(source)

# Afficher les résultats
# print("Code C :")
# print(terme_source_c)
# print("\nCode Fortran :")
# print(terme_source_fortran)
print("\n Code LaTeX source:")
print(equation_latex)

print("\n Code LaTeX C_t:")
latex_Ct = sp.latex(C_t)
print(latex_Ct)

print("\n Code LaTeX C_r:")
latex_Cr = sp.latex(C_r)
print(latex_Cr)

print("\n Code LaTeX C_rr:")
latex_Crr = sp.latex(C_rr)
print(latex_Crr)


