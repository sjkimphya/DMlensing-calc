import numpy as np
from numpy import sqrt, sin, cos, arcsin, arccos, arctan, abs, inner, cross, log10, exp, log
from scipy.special import loggamma
from mpmath import hyp1f1

import random

import ray
from os import cpu_count
cpu_num = cpu_count()   #number of processor
# ray.init(num_cpus=12, ignore_reinit_error=True)

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


def amp_temp(v_vec, m ):    # m, v : natural unit
    r_vec = np.array( [0,0,1] ) * AU/c
    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)

    M = m / hbar
    B = M_sN*M / v
    Y = M * (v*r - np.inner(v_vec, r_vec))

    hyp0 = hyp1f1( 1j*B , 1 , 1j*Y )
    hyp0 = float(hyp0.real) + 1j*float(hyp0.imag)

    amp = exp(pi*B/2 + loggamma(1-1j*B)) * hyp0
    return amp


class DMobData():
    def __init__(self):
        self.t = None
        self.x_vec = None
        self.ob = None
        self.ob_num = None

    def AddData(self, t_data, x_vec_data, ob_data, ob_num):
        data_num = len(t_data)

        if self.ob_num is None:
            self.t = t_data
            self.x_vec = x_vec_data
            self.ob = ob_data
            self.ob_num = np.repeat(ob_num, data_num)
        else:
            self.t = np.concatenate((self.t, t_data), axis=0)
            self.x_vec = np.concatenate((self.x_vec, x_vec_data), axis=0)
            self.ob = np.concatenate((self.ob, ob_data), axis=0)
            self.ob_num = np.concatenate((self.ob_num, np.repeat(ob_num, data_num)), axis=0)


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

    basis_comp_nums = cos(ex1) * sqrt(2/k_num)
    
    if NO_LENSING:
        ob_data_no_lens = np.sum(basis_comp_nums, axis=0)

    if INCLUDE_LENSING:
        pass

    result = DMobData()
    result.AddData(t_data=t_data, x_vec_data=x_data, ob_data=ob_data_no_lens, ob_num=ob_num)

    return result



class stochasticDM():
    def __init__(self, data=None):   # data: DMobData form, seed: integer
        
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

    
    def Log_likelihood(self):  # data_vec: axion field value data vector ; # Cov_mat: covariance matrix, should be numpy array
        
        ob_data_vec = self.data.ob
        cov_mat = self.cov_mat

        # Calculate determinant of Covariance matrix
        # if len(cov_mat.shape)!=2: raise Exception("cov_mat should be n by n matrix.")
        # elif cov_mat.shape[0]!=cov_mat.shape[1]: raise Exception("cov_mat should be n by n matrix.")
        # elif cov_mat.shape[0]!=len(ob_data_vec): raise Exception("cov_mat and data_vec should have the same dimension.")

        
        # dat_num  = len(ob_data_vec)
        # det_cov  = np.linalg.det(cov_mat)
        # if det_cov==0: raise Exception("det_cov is zero.")

        data_mat = np.reshape(ob_data_vec, (-1,1))     # Make data_vec in a colume vector (matrix)
        
        exponent = -0.5 * ( data_mat.T @ np.linalg.inv(cov_mat) @ data_mat).item()
        
        # _likelihood = ((2*pi)**dat_num * det_cov)**-0.5
        # _likelihood = _likelihood * exp(exponent)
        # Log_like = -.5*( dat_num*log(2*pi) + log(abs(det_cov)) ) + exponent
        # _likelihood = exp(Log_like)
        # print(Log_like)
        return -2*exponent#Log_like
    
    def Cor_func(self, dt, dx_vec, v_ob_vec): # <SS'>
        
        g_eff = self.g_eff
        rho   = self.rho_a
        m     = self.m_a
        sig_v = self.sig_v
        # v_ob_vec       (numpy, 3d vector)
        # dt
        # dx_vec (numpy, 3d vector)
        dx2 = np.linalg.norm(dx_vec)**2
        v_ob2 = np.linalg.norm(v_ob_vec)**2
        
        # delta = -.5 * m * dx2/dt
        xi = m * sig_v**2 * dt
        zeta = 1 - 1j*xi

        Const = 1  #g_eff**2 * rho / m**2        #### temp value ####

        # if dt==0:
        #     return Const * exp(-.5* m**2 * sig_v**2 * dx2) * cos(m*np.inner(v_ob_vec, dx_vec))
        # v1_vec = v_ob_vec - (dx_vec / dt)
        
        # ex1  = -1j * m * dt * (1 + .5 * (dx2/dt**2))
        # ex2  = np.inner(v1_vec, v1_vec) / (2*sig_v**2) * (1/zeta - 1)
        # ex1, ex2 each diverge when dt-->0... remove diverging term in ex
        ex = (
            1j * (-m*dt + m/(2*zeta) * (v_ob2*dt - 2*np.inner(v_ob_vec, dx_vec) + 1j*m*sig_v**2*dx2) )
        )
        comp = exp(ex) * zeta**-1.5
        
        result = Const * comp.real
        return result
    


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

        cov_mat = _Cov_matrix(t_data, x_data, v_ob_vec, m, sig_v)

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
def _Cov_matrix(t_data, x_data, v_ob_vec, m, sig_v):
    tot_num = len(t_data)
    set_num_max = (tot_num+1) // 2
    
    t_data_ref = ray.put(t_data)
    x_data_ref = ray.put(x_data)

    result_refs = []
    for i in range(set_num_max):
        testmat = _Cov_mat_process.remote(i, t_data_ref, x_data_ref, v_ob_vec, m, sig_v)
        result_refs.append(testmat)

    Cov_mat = np.zeros((tot_num , tot_num))

    while result_refs:
        done, result_refs = ray.wait(result_refs)
        result = ray.get(done[0])
        _i = result[0]
        temp_arr = result[1]
        for j, val in enumerate(temp_arr):
            if j<_i:
                Cov_mat[_i, j] = val
                Cov_mat[j,_i] = val
            else:
                Cov_mat[tot_num-1-_i, tot_num-1-j] = val
                Cov_mat[tot_num-1-j, tot_num-1-_i] = val

    for i in range(tot_num):
        Cov_mat[i,i] = 1

    return Cov_mat


@ray.remote
def _Cov_mat_process(set_num, t_data, x_data, v_ob_vec, m, sig_v):
    tot_num = len(t_data)
    result = np.zeros(tot_num)
    _i = set_num
    for _j in range(tot_num):
        if _j>_i:
            i = tot_num-1-_i
            j = tot_num-1-_j
        else:
            i = _i
            j = _j

        dt = t_data[i]- t_data[j]
        dx_vec = x_data[i] - x_data[j] 
        result[_j] = _Cov_mat_component(dt,dx_vec, v_ob_vec, m, sig_v)

    return (_i, result)



def _Cov_mat_component(dt, dx_vec, v_ob_vec, m, sig_v): # <SS'>
        
    # v_ob_vec       (numpy, 3d vector)
    
    # dx_vec (numpy, 3d vector) ( [dx, dy, dz] )
    dx2 = np.linalg.norm(dx_vec)**2
    v_ob2 = np.linalg.norm(v_ob_vec)**2
    
    xi = m * sig_v**2 * dt
    zeta = 1 - 1j*xi

    Const = 1  #g_eff**2 * rho / m**2        #### temp value ####
    ex = (
        1j * (-m*dt + m/(2*zeta) * (v_ob2*dt - 2*np.inner(v_ob_vec, dx_vec) + 1j*m*sig_v**2*dx2) )
    )
    comp = exp(ex) * zeta**-1.5
    
    result = Const * comp.real
    return result





if __name__=="__main__":
    ray.init(num_cpus=8, ignore_reinit_error=True)

    ray.shutdown()