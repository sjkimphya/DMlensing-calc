import numpy as np
from numpy import sqrt, sin, cos, arcsin, arccos, arctan, abs, inner, cross, log10, exp
from mpmath import hyp1f1

def amp_point(m, xs, zs, v=1):
    
    r_s = np.sqrt(xs**2 + zs**2)
    v_s = np.sqrt( v**2 )# + 2*G*M_s /(r_s*c)/c**2 )
    a_s = (1+v_s**2)/(2*v_s**2)
    f_s = (m * eV/(2*pi*hbar)) / sqrt(2*(a_s-1))
    
#     print(v_s/v)
    
    X = a_s * 4*M_sN * (pi*f_s)    # re^2/rf^2
    Y = (m * eV/(hbar)) * v * r_s * (1-zs/r_s)

    F = np.exp(pi*X/2 + loggamma(1-1j*X) ) * hyp1f1(1j*X, 1, 1j*Y)
    
    return F