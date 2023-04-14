import numpy as np
from numpy import sqrt, sin, cos, arcsin, arccos, arctan, abs, inner, cross, log10, exp, log
from scipy.special import loggamma
from mpmath import hyp1f1
import mpmath as mp
mp.mp.dps=30
mp.mp.pretty=True

mp2 = mp.mp.clone()
mp2.dps = 12

import random

import ray
from os import cpu_count
cpu_num = cpu_count()   #number of processor
# ray.init(num_cpus=12, ignore_reinit_error=True)

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

mpl.rcParams['text.usetex'] = True
mpl_font = {'size':14, 'family':'serif', 'serif':['Computer Modern']}#, 'serif': ['cm']}
mpl.rc('font', **mpl_font)

from scipy.stats import chi2, norm

#Constants
pi = np.pi
c = 299792458
hbar = 6.626e-34/(2*pi) #SI
m_e = 9.11e-31    # SI
G = 6.67430E-11   #Gravitational constant
G_N = G/c**5      #Natural unit

R_s = 6.9598E+8    #radius of the Sun - BS2005
R_sN = R_s/c
M_s = 1.989e+30   #SI
# M_sN = M_s / (1.3466e+27)/c    # length/time
M_sN = M_s * G/c**3

AU = 149597870700
eV = 1.60217646E-19
ly = 365*24*60*60
Mpc = 3.26156e+6*ly
Gpc = 1000*Mpc
hour = 3600
minute = 60

Error_here = False
Error_code = 1234567

##############################################

@ray.remote
def ray_Amp_Sun(x_idx, v_vecs, r_vecs, m=1e-13*eV/hbar ):
    v_tot_num = len(v_vecs)
    result_dic = {}
    for v_idx in range(v_tot_num):
        v_vec = v_vecs[v_idx]
        r_vec = r_vecs[x_idx]
        result_dic[v_idx] = Amp_Sun(v_vec, r_vec, m)
    return (result_dic, x_idx)

def Amp_Sun(v_vec, r_vec, m=1e-13*eV/hbar ):    # m, v : natural unit


    r_vec = np.array( [0,0,1] ) * AU/c
    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)

    M = m
    B = M_sN*M / v
    Y = M * (v*r - np.inner(v_vec, r_vec))

    hyp0 = mp2.hyp1f1( 1j*B , 1 , 1j*Y )
    hyp0 = float(hyp0.real) + 1j*float(hyp0.imag)

    amp = exp(pi*B/2 + loggamma(1-1j*B)) * hyp0
    
    return amp



class DMobData():
    def __init__(self):
        self.t = None
        self.x_vec = None
        self.ob = None
        self.ob_num = None

    def AddData(self, t_data, x_vec_data, ob_data, ob_num, ob_lensed_data=None):
        data_num = len(t_data)

        if self.ob_num is None:
            self.t = t_data
            self.x_vec = x_vec_data
            self.ob = ob_data
            self.ob_num = np.repeat(ob_num, data_num)
            if ob_lensed_data is not None:
                self.ob_lensed = ob_lensed_data
        else:
            self.t = np.concatenate((self.t, t_data), axis=0)
            self.x_vec = np.concatenate((self.x_vec, x_vec_data), axis=0)
            self.ob = np.concatenate((self.ob, ob_data), axis=0)
            self.ob_num = np.concatenate((self.ob_num, np.repeat(ob_num, data_num)), axis=0)
            if ob_lensed_data is not None:
                self.ob_lensed = np.concatenate((self.ob_lensed, ob_lensed_data), axis=0)


# class DMobData():
#     def __init(self):
#         self.t = {}
#         self.x_vec = {}
#         self.ob = {}

#     def AddData(self, t_data, x_vec_data, ob_data, ob_num): # data --> numpy array
#         self.t[ob_num] = t_data
#         self.x_vec[ob_num] = x_vec_data
#         self.ob[ob_num] = ob_data

#     def Out(self, type=None):   # type = t, x, ob
#         if type=="t":
#             return np.concatenate([self.t[j] for j in sorted(self.t)], axis=0)
#         if type=="x":
#             return np.concatenate([self.x_vec[j] for j in sorted(self.x_vec)], axis=0)
#         if type=="ob":
#             return np.concatenate([self.ob[j] for j in sorted(self.ob)], axis=0)
        

def DMRandomGenerator(t_data, x_data, NO_LENSING=True, INCLUDE_LENSING=False, v0_vec = None
                      , sig_v = None, m=1e-13*eV/hbar, seed=None, k_num = 1000, ob_num=0):
    # t_data: time series numpy ,    x_data: location series 3-vectors numpy array   
    # v0_vec: v0 vector
    if v0_vec is None:
        _v0_vec = np.array( [0,0,8e-4] )
    else:
        _v0_vec = v0_vec
    # sig_v: std.dv. of v-vectors
    if sig_v is None:
        _sig_v = np.linalg.norm(_v0_vec) / sqrt(2)
    else:
        _sig_v = sig_v

    # Random generator
    data_len = len(t_data)
    _rng = np.random.default_rng(seed)
    rnd_phase = np.tile(_rng.random(k_num), (data_len, 1)).T * (2*pi)      # random phase \in [0, 2*pi]

    # Generate random k
    _temp = {}
    for idx in range(3):
        _temp[idx] = _rng.normal(_v0_vec[idx], _sig_v, size=(k_num,1))
    v_vecs = np.concatenate( [_temp[j] for j in sorted(_temp)], axis=1 )
    omegas = sqrt(np.linalg.norm(v_vecs, axis=1)**2 + 1) * m
    ex1 =  np.tensordot(omegas, t_data, axes=0) - m*np.tensordot(v_vecs, x_data.T, axes=1) + rnd_phase  # (k_num*dat_num) array

    basis_comp_nums = np.exp(1j*ex1) * sqrt(2/k_num)

    if NO_LENSING:
        ob_data_no_lens = np.sum(basis_comp_nums.real, axis=0)

    if INCLUDE_LENSING:
        t_num = len(t_data)
        amps = np.zeros((k_num, t_num), dtype=complex)

        # not use ray
        # for v_idx in range(k_num):
        #     for x_idx in range(t_num):  
        #         amps[v_idx, x_idx] = Amp_Sun(v_vecs[v_idx], x_data[x_idx])

        # use ray
        v_ref = ray.put(v_vecs)
        x_ref = ray.put(x_data)

        result_refs = []
        for x_idx in range(t_num):
            each_amp_ref = ray_Amp_Sun.remote(x_idx, v_ref, x_ref, m)
            result_refs.append(each_amp_ref)

        while result_refs:
            done, result_refs = ray.wait(result_refs)
            result_pack = ray.get(done[0])
            amp_result = result_pack[0]
            x_idx = result_pack[1]
            
            for v_idx in range(k_num):
                amps[v_idx, x_idx] = amp_result[v_idx]
        #

        lensed_comp_nums = basis_comp_nums * amps
        ob_lensed_data = np.sum(lensed_comp_nums.real, axis=0)
    else:
        ob_lensed_data = None

    result = DMobData()
    result.AddData(t_data=t_data, x_vec_data=x_data, ob_data=ob_data_no_lens, ob_lensed_data = ob_lensed_data, ob_num=ob_num)

    return result



class stochasticDM():
    def __init__(self, data=None, mpmath=True):   # data: DMobData class, mpmath: whether using mpmath package for calc.
        
        self.mpmath = mpmath
        # # Random generator
        # self.seed = seed
        # self._rng = random.Random(seed)
        # self.rnd_phase = self._rng.random() * 2*pi      # random phase \in [0, 2*pi]

        # environment constant
        self.g_eff = 1
        self.rho_a = 1
        self.m_a = 1e-13 * eV / hbar
        self.v0_vec = np.array( [0,0,8e-4] )
        self.sig_v = np.linalg.norm(self.v0_vec) / sqrt(2)
        

        if data is not None:
            self.data = data

        # Covariance matrix
        self.cov_mat = self.Cov_matrix()
        if self.mpmath:
            self.inv_cov_mat = self.cov_mat**-1
        else:
            self.inv_cov_mat = np.linalg.inv(self.cov_mat)

    
    def Log_likelihood(self, lensed=False):  # data_vec: axion field value data vector ; # Cov_mat: covariance matrix, should be numpy array
        
        ob_data_vec = self.data.ob
        cov_mat = self.cov_mat

        # Calculate determinant of Covariance matrix
        # if len(cov_mat.shape)!=2: raise Exception("cov_mat should be n by n matrix.")
        # elif cov_mat.shape[0]!=cov_mat.shape[1]: raise Exception("cov_mat should be n by n matrix.")
        # elif cov_mat.shape[0]!=len(ob_data_vec): raise Exception("cov_mat and data_vec should have the same dimension.")

        
        # dat_num  = len(ob_data_vec)
        # det_cov  = np.linalg.det(cov_mat)
        # if det_cov==0: raise Exception("det_cov is zero.")
        if self.mpmath:
            if lensed:
                data_mat = mp.matrix(self.data.ob_lensed)
            else:
                data_mat = mp.matrix(ob_data_vec)     # Make data_vec in a colume vector (matrix)
        else:
            if lensed:
                data_mat = np.array(self.data.ob_lensed)
            else:
                data_mat = np.array(ob_data_vec)     # Make data_vec in a colume vector (matrix)

        inv_mat = self.inv_cov_mat

        if self.mpmath: exponent = -0.5 * ( data_mat.T * inv_mat * data_mat)[0]
        else: exponent = -0.5 * ( data_mat.T @ inv_mat @ data_mat)
        
        # _likelihood = ((2*pi)**dat_num * det_cov)**-0.5
        # _likelihood = _likelihood * exp(exponent)
        # Log_like = -.5*( dat_num*log(2*pi) + log(abs(det_cov)) ) + exponent
        # _likelihood = exp(Log_like)
        # print(Log_like)
        return -2*exponent #Log_like
    
    # def Cor_func(self, dt, dx_vec, v_ob_vec): # <SS'>
        
    #     g_eff = self.g_eff
    #     rho   = self.rho_a
    #     m     = self.m_a
    #     sig_v = self.sig_v
    #     # v_ob_vec       (numpy, 3d vector)
    #     # dt
    #     # dx_vec (numpy, 3d vector)
    #     dx2 = np.linalg.norm(dx_vec)**2
    #     v_ob2 = np.linalg.norm(v_ob_vec)**2
        
    #     # delta = -.5 * m * dx2/dt
    #     xi = m * sig_v**2 * dt
    #     zeta = 1 - 1j*xi

    #     Const = 1  #g_eff**2 * rho / m**2        #### temp value ####

    #     # if dt==0:
    #     #     return Const * exp(-.5* m**2 * sig_v**2 * dx2) * cos(m*np.inner(v_ob_vec, dx_vec))
    #     # v1_vec = v_ob_vec - (dx_vec / dt)
        
    #     # ex1  = -1j * m * dt * (1 + .5 * (dx2/dt**2))
    #     # ex2  = np.inner(v1_vec, v1_vec) / (2*sig_v**2) * (1/zeta - 1)
    #     # ex1, ex2 each diverge when dt-->0... remove diverging term in ex
    #     ex = (
    #         1j * (-m*dt + m/(2*zeta) * (v_ob2*dt - 2*np.inner(v_ob_vec, dx_vec) + 1j*m*sig_v**2*dx2) )
    #     )
    #     comp = exp(ex) * zeta**-1.5
        
    #     result = Const * comp.real
    #     return result
    


    def Cov_matrix(self):
        t_data = self.data.t
        x_data = self.data.x_vec
        v_ob_vec = self.v0_vec  # temporary... should be corrected
        m = self.m_a
        sig_v = self.sig_v

        # tot_num = len(t_data)
        # cov_mat = np.zeros((tot_num, tot_num))

        # dt_mat = t_data[:,None] - t_data[None,:]
        # dx_mat = x_data[:,None,:] - x_data[None,:,:]

        # for i in range(tot_num):
        #     for j in range(tot_num):
        #         dt = t_data[i] - t_data[j]
        #         dx_vec = x_data[i] - x_data[j]
        #         v_ob_vec = self.v0_vec  # temporary... should be corrected

        #         cov_mat[i,j] = self.Cor_func(dt, dx_vec, v_ob_vec)

        cov_mat = _Cov_matrix(t_data, x_data, v_ob_vec, m, sig_v, self.mpmath)

        # __cor_func = np.vectorize(self.__temp_cor_func)
        # cov_mat = __cor_func(range(tot_num), range(tot_num))

        return cov_mat
    
    # def __temp_cor_func(self, i, j):
    #     t_data = self.data.t
    #     x_data = self.data.x_vec

    #     # tot_num = len(t_data)

    #     dt = t_data[i] - t_data[j]
    #     dx_vec = x_data[i] - x_data[j]
    #     v_ob_vec = self.v0_vec

    #     return self.Cor_func(dt, dx_vec, v_ob_vec)



### Multiprocessing for correlation function calculation
def _Cov_matrix(t_data, x_data, v_ob_vec, m, sig_v, mpmath=True):
    tot_num = len(t_data)
    set_num_max = (tot_num+1) // 2
    
    t_data_ref = ray.put(t_data)
    x_data_ref = ray.put(x_data)

    # Cov_mat = np.zeros((tot_num , tot_num))
    if mpmath:
        Cov_mat = mp.matrix(tot_num)
    else:
        Cov_mat = np.zeros((tot_num, tot_num))

    result_refs = []
    for i in range(set_num_max):
        testmat = _Cov_mat_process.remote(i, t_data_ref, x_data_ref, v_ob_vec, m, sig_v, mpmath)
        result_refs.append(testmat)

    while result_refs:
        done, result_refs = ray.wait(result_refs)
        result = ray.get(done[0])
        # print(ray.get(result))   ######################################3
        _i = result[0]
        temp_dic = result[1]
        for j in range(tot_num):
            val = temp_dic[j]
            if j<_i:
                Cov_mat[_i, j] = val
                Cov_mat[j,_i] = val
            else:
                Cov_mat[tot_num-1-_i, tot_num-1-j] = val
                Cov_mat[tot_num-1-j, tot_num-1-_i] = val

    for i in range(tot_num):
        if mpmath: Cov_mat[i,i] = mp.mpf('1')
        else: Cov_mat[i,i] = 1

    return Cov_mat

@ray.remote
def _Cov_mat_process(set_num, t_data, x_data, v_ob_vec, m, sig_v, mpmath=True):
    return Cov_mat_process(set_num, t_data, x_data, v_ob_vec, m, sig_v, mpmath)

def Cov_mat_process(set_num, t_data, x_data, v_ob_vec, m, sig_v, mpmath=True):
    tot_num = len(t_data)
    # result = mp.zeros(1,tot_num)
    result = {}
    # result = np.zeros(tot_num)
    _i = set_num
    for _j in range(tot_num):
        if _j>_i:
            i = tot_num-1-_i
            j = tot_num-1-_j
        else:
            i = _i
            j = _j
        if mpmath:
            dt = mp.mpf(t_data[i]) - mp.mpf(t_data[j])
            dx_vec = mp.matrix(x_data[i]) - mp.matrix(x_data[j])
        else:
            dt = t_data[i] - t_data[j]
            dx_vec = x_data[i] - x_data[j]
        result[_j] = _Cov_mat_component(dt, dx_vec, v_ob_vec, m, sig_v, mpmath)
        # nd_result = np.array(result.tolist(),dtype=np.float64)
        # re = mp.mpf('1')
    return (_i, result)


def Cov_mat_component(dt, dx_vec, v_ob_vec, m, sig_v, mpmath=True): # <SS'>
    return _Cov_mat_component(dt, dx_vec, v_ob_vec, m, sig_v, mpmath)

def _Cov_mat_component(dt, dx_vec, v_ob_vec, m, sig_v, mpmath=True): # <SS'>
        
    # v_ob_vec       (numpy, 3d vector)
    # dx_vec (numpy, 3d vector) ( [dx, dy, dz] )
    if mpmath:
        _v_ob_vec = mp.matrix(v_ob_vec)
        v_ob2 = mp.norm(_v_ob_vec)**2
        _j = mp.mpc(1j)

        _m = mp.mpf(m)
        _sig_v = mp.mpf(sig_v)
        _dt = mp.mpf(dt)
        _dx_vec = mp.matrix(dx_vec)
    else:
        _v_ob_vec = v_ob_vec
        v_ob2 = np.linalg.norm(_v_ob_vec)**2
        _j = 1j

        _m = m
        _sig_v = sig_v
        _dt = dt
        _dx_vec = dx_vec

    xi = _m * _sig_v**2 * _dt
    zeta = 1 - _j*xi
    chi_vec = (_v_ob_vec - _j * _m * _sig_v**2 * _dx_vec)

    if mpmath: chi2 = (chi_vec.T * chi_vec)[0]
    else: chi2 = ( chi_vec.T @ chi_vec ).item()

    Const = 1  #g_eff**2 * rho / m**2        #### temp value ####
    # ex = (
    #     1j * (+m*dt + m/(2*zeta) * (v_ob2*dt - 2*np.inner(v_ob_vec, dx_vec) + 1j*m*sig_v**2*dx2) )
    # )
    ex = (
        (chi2/zeta - v_ob2) / (2 * _sig_v**2) + _j * _m * _dt
    )
    if mpmath: comp = mp.exp(ex) * zeta**-1.5
    else:      comp = exp(ex) * zeta**-1.5
    
    result = Const * comp.real
    return result


if __name__=="__main__":
    pass

