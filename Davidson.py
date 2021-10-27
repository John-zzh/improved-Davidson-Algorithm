#!/usr/bin/python

import time
import numpy as np
import mathlib
import parameterlib
from opt_einsum import contract as einsum
# import pyscf
from pyscf import gto, scf, dft, tddft, data, lib
import argparse
import os, sys
import psutil
import yaml

'''wb97x  methanol, 1e-5
  sTDDFT no truncate [6.46636611 8.18031534 8.38140651 9.45011415 9.5061059 ]
            40 eV    [6.46746642 8.18218267 8.38314651 9.45214869 9.5126739 ]
    sTDA no truncate [6.46739711 8.18182208 8.38358473 9.45195554 9.52133129]
            40 eV    [6.46827111 8.18334703 8.38483801 9.45361525 9.52562255]
'''
print('curpath', os.getcwd())
print('lib.num_threads() = ', lib.num_threads())

def str2bool(str):
    if str.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif str.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

def gen_args():
    parser = argparse.ArgumentParser(description='Davidson')
    parser.add_argument('-x', '--xyzfile',          type=str,   default='methanol.xyz',  help='xyz filename (molecule.xyz)')
    parser.add_argument('-chk', '--checkfile',      type=str2bool,  default=True, help='checkpoint filename (.chk)')
    parser.add_argument('-m', '--method',           type=str,   default='RKS', help='RHF RKS UHF UKS')
    parser.add_argument('-f', '--functional',       type=str,   default='pbe0',  help='xc functional')
    parser.add_argument('-b', '--basis_set',        type=str,   default='def2-svp',  help='basis set')
    parser.add_argument('-df', '--density_fit',     type=str2bool,  default=True,  help='density fitting turn on')
    parser.add_argument('-g', '--grid_level',       type=int,   default='3',   help='0-9, 9 is best')

    parser.add_argument('-n','--nstates',           type=int,   default = 5,      help='number of excited states')
    parser.add_argument('-pytd','--pytd',           type=str2bool,  default = False , help='whether to compare with PySCF TDDFT')

    parser.add_argument('-TDA','--TDA',             type=str2bool,  default = False, help='perform TDA')
    parser.add_argument('-TDDFT','--TDDFT',         type=str2bool,  default = False, help='perform TDDFT')
    parser.add_argument('-dynpol','--dynpol',       type=str2bool,  default = False, help='perform dynamic polarizability')
    parser.add_argument('-omega','--dynpol_omega',  type=float, default = [], nargs='+', help='dynamic polarizability with perurbation omega, a list')
    parser.add_argument('-stapol','--stapol',       type=str2bool,  default = False, help='perform static polarizability')
    parser.add_argument('-sTDA','--sTDA',           type=str2bool,  default = False, help='perform sTDA calculation')
    parser.add_argument('-sTDDFT','--sTDDFT',       type=str2bool,  default = False, help='perform sTDDFT calculation')
    parser.add_argument('-TT','--Truncate_test',    type=str2bool,  default = False, help='test the wall time for different virtual truncation')
    parser.add_argument('-tr','--traceAA',          type=str2bool,  default = False, help='test different tr([A,A])')
    parser.add_argument('-AV','--AV',               type=str2bool,  default = False, help='AV = A*V or lambda V')

    parser.add_argument('-TV','--truncate_virtual', type=float, default = 40,    help='the threshold to truncate virtual orbitals, in eV')

    parser.add_argument('-o','--ip_options',        type=int,   default = [0], nargs='+', help='0-7')
    parser.add_argument('-t','--conv_tolerance',    type=float, default= 1e-5, help='residual norm Convergence threhsold')

    parser.add_argument('-it','--initial_TOL',      type=float, default= 1e-3, help='conv for the inital guess')
    parser.add_argument('-pt','--precond_TOL',      type=float, default= 1e-2, help='conv for TDA preconditioner')

    parser.add_argument('-ei','--extrainitial',     type=int,   default= 8,    help='number of extral TDA initial guess vectors, 0-8')
    parser.add_argument('-max','--max',             type=int,   default= 35,   help='max iterations')
    parser.add_argument('-DK','--DKapp',            type=str2bool,  default= False, help='K = (A^se - λ)^-1 or K = (diag(A) - λ)^-1')

    parser.add_argument('-et','--eigensolver_tol',  type=float, default= 1e-5, help='conv for new guess generator in new_ES')
    parser.add_argument('-M','--memory',            type=int,   default= 4000, help='max_memory')
    parser.add_argument('-v','--verbose',           type=int,   default= 3,    help='mol.verbose = 3,4,5')

    parser.add_argument('-beta','--beta',           type=float, default= None,  help='beta = 4.00')
    parser.add_argument('-alpha','--alpha',         type=float, default= None,  help='alpha = 0.83')

    parser.add_argument('-tuning','--tuning',       type=str2bool, default= False,    help='turn on on-the-fly tuning')

    parser.add_argument('-beta_list','--beta_list',   type=float, default= [],    nargs='+', help='8 7 6 5 4 3 2')
    parser.add_argument('-alpha_list','--alpha_list', type=float, default= [],    nargs='+', help='8 7 6 5 4 3 2 1.8 1 0.8')

    args = parser.parse_args()
    if args.dynpol == True and args.dynpol_omega == []:
        raise ValueError('External Perturbation ω cannot be None')
    return args
args = gen_args()

def show_memory_info(hint):
    pid = os.getpid()
    p = psutil.Process(pid)
    info = p.memory_full_info()
    memory = info.uss / 1024**3
    print('{} memory used: {:<.2f} GB'.format(hint, memory))

show_memory_info('at beginning')

basename = args.xyzfile.split('.',1)[0]

def SCF_kernel(xyzfile=args.xyzfile, mol=gto.Mole(), basis_set=args.basis_set,
                                    verbose=args.verbose, max_memory=args.memory,
                                    method=args.method, checkfile=args.checkfile,
                                    density_fit=args.density_fit):
    kernel_0 = time.time()

    '''read xyz file and delete its first two lines'''
    f = open(xyzfile)
    atom_coordinates = f.readlines()
    del atom_coordinates[:2]

    '''build geometry in PySCF'''

    mol.atom = atom_coordinates
    mol.basis = args.basis_set
    mol.verbose = args.verbose
    mol.max_memory = args.memory
    print('mol.max_memory', mol.max_memory)
    mol.build(parse_arg = False)

    '''DFT or HF'''
    if method == 'RKS':
        mf = dft.RKS(mol)
    elif method == 'UKS':
        mf = dft.UKS(mol)
    elif method == 'RHF':
        mf = scf.RHF(mol)
    elif method == 'UHF':
        mf = scf.UHF(mol)

    if 'KS' in args.method:
        print('RKS')
        mf.xc = args.functional
        mf.grids.level = args.grid_level
    else:
        print('HF')
    if density_fit:
        mf = mf.density_fit()
        print('Density fitting turned on')
    if checkfile == True:
        '''use the *.chk file as scf input'''
        mf.chkfile = basename + '_' + args.functional + '.chk'
        mf.init_guess = 'chkfile'
    mf.conv_tol = 1e-10
    print ('Molecule built')
    print ('Calculating SCF Energy...')
    mf.kernel()

    kernel_1 = time.time()
    kernel_t = kernel_1 - kernel_0
    print ('SCF Done after %.2f'%kernel_t, 'seconds')

    return mol, mf, kernel_t

mol, mf, kernel_t = SCF_kernel()

show_memory_info('after SCF')

'''Collect everything needed from PySCF'''

def gen_global_var(tddft, mf, mol, functional):
    '''TDA_vind & TDDFT_vind are ab-initio matrix vector multiplication function
    '''
    td = tddft.TDA(mf)
    TD = tddft.TDDFT(mf)
    '''hdiag is one dinension matrix, (A_size,)
    '''
    TDA_vind, hdiag = td.gen_vind(mf)
    TDDFT_vind, Hdiag = TD.gen_vind(mf)

    Natm = mol.natm
    mo_occ = mf.mo_occ
    '''mf.mo_occ is an array of occupance [2,2,2,2,2,0,0,0,0.....]
       N_bf is the total amount of MOs
       if no truncation, then max_vir = n_vir and n_occ + max_vir = N_bf
    '''

    ''' produce orthonormalized coefficient matrix C, N_bf * N_bf
        mf.mo_coeff is the unorthonormalized coefficient matrix
        S = mf.get_ovlp()  is basis overlap matrix
        S = np.dot(np.linalg.inv(c.T), np.linalg.inv(c))

        C_matrix is the orthonormalized coefficient matrix
        np.dot(C_matrix.T,C_matrix) is a an identity matrix
    '''
    S = mf.get_ovlp()
    X = mathlib.matrix_power(S, 0.5)
    C_matrix = np.dot(X,mf.mo_coeff)

    N_bf = len(mo_occ)
    n_occ = len(np.where(mo_occ > 0)[0])
    n_vir = len(np.where(mo_occ == 0)[0])
    delta_hdiag = hdiag.reshape(n_occ, n_vir)
    A_size = n_occ * n_vir

    tol_eV = args.truncate_virtual/parameterlib.Hartree_to_eV
    homo_vir = delta_hdiag[-1,:]
    max_vir = len(np.where(homo_vir <= tol_eV)[0])

    max_vir_hdiag = delta_hdiag[:,:max_vir]
    rst_vir_hdiag = delta_hdiag[:,max_vir:]

    A_reduced_size = n_occ * max_vir

    '''R_array is inter-particle distance array
       unit == ’Bohr’, 5.29177210903(80)×10^(−11) m
    '''
    R_array = gto.mole.inter_distance(mol, coords=None)

    a_x, beta, alpha = parameterlib.gen_alpha_beta_ax(functional)

    if args.beta != None:
        beta = args.beta
        alpha = args.alpha

    print('a_x =', a_x)
    print('beta =', beta)
    print('alpha =', alpha)

    print('n_occ = ', n_occ)
    print('n_vir = ', n_vir)
    print('max_vir = ', max_vir)
    print('A_size = ', A_size)
    print('A_reduced_size =', A_reduced_size)

    return (TDA_vind, TDDFT_vind, hdiag, delta_hdiag, max_vir_hdiag,
            rst_vir_hdiag, Natm, C_matrix, N_bf, n_occ, n_vir, max_vir, A_size,
            A_reduced_size, R_array, a_x, beta, alpha)

(TDA_vind, TDDFT_vind, hdiag, delta_hdiag, max_vir_hdiag, rst_vir_hdiag, Natm,
C_matrix, N_bf, n_occ, n_vir, max_vir, A_size, A_reduced_size, R_array, a_x,
beta, alpha) = gen_global_var( tddft, mf, mol, functional=args.functional)

def TDA_matrix_vector(V):
    '''return AX'''
    return TDA_vind(V.T).T

def TDDFT_matrix_vector(X, Y):
    '''return AX + BY and AY + BX'''
    XY = np.vstack((X,Y)).T
    U = TDDFT_vind(XY)
    U1 = U[:,:A_size].T
    U2 = -U[:,A_size:].T
    return U1, U2

def static_polarizability_matrix_vector(X):
    '''return (A+B)X
       this is not the optimum way, but the only way in PySCF
    '''
    U1, U2 = TDDFT_matrix_vector(X,X)
    return U1

def gen_gammaJK(mol=mol, Natm=Natm, R_array=R_array, beta=beta, alpha=alpha):
    '''creat GammaK and GammaK matrix
       mol.atom_pure_symbol(atom_id) returns the element symbol
    '''
    HARDNESS = parameterlib.gen_HARDNESS()
    '''a list is a list of chemical hardness for all atoms
    '''
    a = [HARDNESS[mol.atom_pure_symbol(atom_id)] for atom_id in range(Natm)]
    a = np.asarray(a).reshape(1,-1)
    eta = (a+a.T)/2
    GammaJ = (R_array**beta + (a_x * eta)**(-beta))**(-1/beta)
    GammaK = (R_array**alpha + eta**(-alpha)) **(-1/alpha)

    # print(R_array)
    # print(eta)
    # print(1/R_array)
    # print(GammaJ)
    return GammaJ, GammaK

GammaJ, GammaK = gen_gammaJK(alpha=alpha, beta=beta)

def gen_QJK(C_matrix=C_matrix, mol=mol, Natm=Natm, N_bf=N_bf, max_vir=max_vir,
                                                GammaJ=GammaJ, GammaK=GammaK):
    # Qstart = time.time()
    '''build q_iajb tensor'''

    aoslice = mol.aoslice_by_atom()
    q_tensors = np.zeros([Natm, N_bf, N_bf])
    for atom_id in range(Natm):
        shst, shend, atstart, atend = aoslice[atom_id]
        q_tensors[atom_id,:,:] = \
                np.dot(C_matrix[atstart:atend,:].T, C_matrix[atstart:atend,:])

    '''pre-calculate and store the Q-Gamma rank 3 tensor
       qia * gamma * qjb -> qia GK_q_jb
    '''
    q_ij = np.zeros((Natm, n_occ, n_occ))
    q_ij[:,:,:] = q_tensors[:,:n_occ,:n_occ]

    q_ab = np.zeros((Natm, max_vir, max_vir))
    q_ab[:,:,:] = q_tensors[:,n_occ:n_occ+max_vir,n_occ:n_occ+max_vir]

    q_ia = np.zeros((Natm, n_occ, max_vir))
    q_ia[:,:,:] = q_tensors[:,:n_occ,n_occ:n_occ+max_vir]

    GK_q_jb = einsum("Bjb,AB->Ajb", q_ia, GammaK)
    GJ_q_ab = einsum("Bab,AB->Aab", q_ab, GammaJ)
    # Qend = time.time()
    #
    # Q_time = Qend - Qstart
    # print('Q-Gamma tensors building time = %.2f'%Q_time)
    # show_memory_info('after Q matrix')
    return q_ij, q_ab, q_ia, GK_q_jb, GJ_q_ab



def gen_iajb_ijab_ibja_delta_fly(max_vir=max_vir, GammaJ=GammaJ, GammaK=GammaK,
                        delta_hdiag=delta_hdiag, max_vir_hdiag=max_vir_hdiag,
                        rst_vir_hdiag=rst_vir_hdiag):
    '''define sTDA on-the-fly two electron intergeral (pq|rs)
       A_iajb * v = delta_ia_ia*v + 2(ia|jb)*v - (ij|ab)*v
       iajb_v = einsum('Aia,Bjb,AB,jbm -> iam', q_ia, q_ia, GammaK, V)
       ijab_v = einsum('Aij,Bab,AB,jbm -> iam', q_ij, q_ab, GammaJ, V)
    '''
    q_ij, q_ab, q_ia, GK_q_jb, GJ_q_ab = gen_QJK(GammaJ=GammaJ, GammaK=GammaK)

    def iajb_fly(V):
        '''(ia|jb) '''
        GK_q_jb_V = einsum("Ajb,jbm->Am", GK_q_jb, V)
        iajb_V = einsum("Aia,Am->iam", q_ia, GK_q_jb_V)
        return iajb_V

    def ijab_fly(V):
        '''(ij|ab) '''
        GJ_q_ab_V = einsum("Aab,jbm->Ajam", GJ_q_ab, V)
        ijab_V = einsum("Aij,Ajam->iam", q_ij, GJ_q_ab_V)
        return ijab_V

    def ibja_fly(V):
        '''the Forck exchange energy in B matrix
           (ib|ja)
        '''
        q_ib_V = einsum("Aib,jbm->Ajim", q_ia, V)
        ibja_V = einsum("Aja,Ajim->iam", GK_q_jb, q_ib_V)
        return ibja_V

    def delta_fly(V):
        '''delta_hdiag.shape = (n_occ, n_vir)'''
        delta_v = einsum("ia,iam->iam", delta_hdiag, V)
        return delta_v

    def delta_max_vir_fly(V):
        '''max_vir_hdiag.shape = (n_occ, max_vir)'''
        delta_max_vir_v = einsum("ia,iam->iam", max_vir_hdiag, V)
        return delta_max_vir_v

    def delta_rst_vir_fly(V):
        '''max_vir_hdiag.shape = (n_occ, n_vir-max_vir)'''
        delta_rst_vir_v = einsum("ia,iam->iam", rst_vir_hdiag, V)
        return delta_rst_vir_v

    return (iajb_fly, ijab_fly, ibja_fly, delta_fly, delta_max_vir_fly,
            delta_rst_vir_fly)

def gen_mv_fly(GammaJ=GammaJ, GammaK=GammaK, n_occ=n_occ, max_vir=max_vir,
                                                            A_size=A_size):

    (iajb_fly, ijab_fly, ibja_fly, delta_fly, delta_max_vir_fly,
    delta_rst_vir_fly) = gen_iajb_ijab_ibja_delta_fly(GammaJ=GammaJ,
                                                      GammaK=GammaK)
    def sTDA_mv(V):
        '''return AX'''
        V = V.reshape(n_occ, max_vir, -1)
        '''MV =  delta_fly(V) + 2*iajb_fly(V) - ijab_fly(V)'''
        MV = delta_max_vir_fly(V) + 2*iajb_fly(V) - ijab_fly(V)
        MV = MV.reshape(n_occ*max_vir,-1)
        return MV

    def full_sTDA_mv(V, sTDA_mv=sTDA_mv, delta_rst_vir_fly=delta_rst_vir_fly):

        V = V.reshape(n_occ,n_vir,-1)
        U = np.zeros_like(V)

        V1 = V[:,:max_vir,:]
        V2 = V[:,max_vir:,:]

        U[:,:max_vir,:] = sTDA_mv(V1).reshape(n_occ, max_vir, -1)
        U[:,max_vir:,:] = delta_rst_vir_fly(V2)

        U = U.reshape(A_size,-1)
        return U

    def sTDDFT_mv(X, Y):
        '''return AX+BY and AY+BX
           sTDA_A =  delta_fly(V) + 2*iajb_fly(V) - ijab_fly(V)
           sTDDFT_B = 2*iajb_fly(V) - a_x*ibja_fly(V)
        '''
        X = X.reshape(n_occ, max_vir,-1)
        Y = Y.reshape(n_occ, max_vir,-1)

        X_max_vir = X[:,:max_vir,:]
        Y_max_vir = Y[:,:max_vir,:]

        iajb_X = iajb_fly(X_max_vir)
        iajb_Y = iajb_fly(Y_max_vir)

        ijab_X = ijab_fly(X_max_vir)
        ijab_Y = ijab_fly(Y_max_vir)

        ibja_X = ibja_fly(X_max_vir)
        ibja_Y = ibja_fly(Y_max_vir)

        delta_X = delta_max_vir_fly(X_max_vir)
        delta_Y = delta_max_vir_fly(Y_max_vir)

        AX = delta_X + 2*iajb_X - ijab_X
        AY = delta_Y + 2*iajb_Y - ijab_Y

        BX = 2*iajb_X - a_x*ibja_X
        BY = 2*iajb_Y - a_x*ibja_Y

        U1 = np.zeros_like(X)
        U2 = np.zeros_like(X)

        U1[:,:max_vir,:] = AX + BY
        U2[:,:max_vir,:] = AY + BX

        U1 = U1.reshape(n_occ*max_vir,-1)
        U2 = U2.reshape(n_occ*max_vir,-1)

        return U1, U2

    def sTDDFT_stapol_mv(X):
        '''return (A+B)X = delta_fly(V) + 4*iajb_fly(V)
           - ijab_fly(V) - a_x*ibja_fly(V)
        '''
        X = X.reshape(n_occ, max_vir, -1)
        U = delta_max_vir_fly(X) + 4*iajb_fly(X) - ijab_fly(X) - a_x*ibja_fly(X)
        U = U.reshape(n_occ*max_vir, -1)

        return U

    return sTDA_mv, full_sTDA_mv, sTDDFT_mv, sTDDFT_stapol_mv

sTDA_mv, full_sTDA_mv, sTDDFT_mv, sTDDFT_stapol_mv = gen_mv_fly(GammaJ=GammaJ,
                                                                GammaK=GammaK)

class on_the_fly_tensors(object):
    ''' iajb_fly,
        ijab_fly,
        ibja_fly,
        delta_fly,
        delta_max_vir_fly,
        delta_rst_vir_fly
    '''
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

        '''first, update the gamma matrix
        '''
        (self.GammaJ,
        self.GammaK) = gen_gammaJK(alpha=self.alpha, beta=self.beta)

        '''then generate tensors
        '''
        (self.iajb_fly,
        self.ijab_fly,
        self.ibja_fly,
        self.delta_fly,
        self.delta_max_vir_fly,
        self.delta_rst_vir_fly) = gen_iajb_ijab_ibja_delta_fly(
                                        GammaJ=self.GammaJ, GammaK=self.GammaK)
    def sTDA_mv(self, V):
        '''return AX'''
        V = V.reshape(n_occ, max_vir, -1)
        '''MV =  delta_fly(V) + 2*iajb_fly(V) - ijab_fly(V)'''
        MV = self.delta_max_vir_fly(V) + 2*self.iajb_fly(V) - self.ijab_fly(V)
        MV = MV.reshape(n_occ*max_vir,-1)
        return MV

    def full_sTDA_mv(self, V):
        V = V.reshape(n_occ,n_vir,-1)
        U = np.zeros_like(V)
        V1 = V[:,:max_vir,:]
        V2 = V[:,max_vir:,:]
        U[:,:max_vir,:] = self.sTDA_mv(V1).reshape(n_occ, max_vir, -1)
        U[:,max_vir:,:] = self.delta_rst_vir_fly(V2)
        U = U.reshape(A_size,-1)
        return U


def TDA_A_diag_initial_guess(m, hdiag=hdiag):
    '''m is the amount of initial guesses'''
    hdiag = hdiag.reshape(-1,)
    V_size = hdiag.shape[0]
    Dsort = hdiag.argsort()
    energies = hdiag[Dsort][:m]*parameterlib.Hartree_to_eV
    V = np.zeros((V_size, m))
    for j in range(m):
        V[Dsort[j], j] = 1.0
    return V, energies

def TDA_A_diag_preconditioner(residual, sub_eigenvalue, hdiag=hdiag,
                            current_dic=None, tol=None, full_guess=None,
                        return_index=None, W_H=None, V_H=None, sub_A_H=None,
                                    matrix_vector_product = None):
    '''DX = XΩ'''
    k = np.shape(residual)[1]
    t = 1e-14
    D = np.repeat(hdiag.reshape(-1,1), k, axis=1) - sub_eigenvalue
    '''force all small values not in [-t,t]'''
    D = np.where( abs(D) < t, np.sign(D)*t, D)
    new_guess = residual/D

    if current_dic != None:
        return new_guess, current_dic
    else:
        return new_guess

def sTDA_eigen_solver(k, tol=args.initial_TOL, matrix_vector_product=sTDA_mv):
    '''A'X = XΩ'''
    print('sTDA nstate =', k)
    sTDA_D_start = time.time()
    max = 30

    '''m is size of subspace'''
    m = 0
    new_m = min([k+8, 2*k, A_size])
    V = np.zeros((A_reduced_size, max*k + m))
    W = np.zeros_like(V)

    '''V is subsapce basis
       W is transformed guess vectors'''
    V[:, :new_m], initial_energies = TDA_A_diag_initial_guess(m=new_m,
                                                          hdiag=max_vir_hdiag)
    for i in range(max):
        '''create subspace'''
        W[:, m:new_m] = matrix_vector_product(V[:, m:new_m])
        sub_A = np.dot(V[:,:new_m].T, W[:,:new_m])
        sub_A = mathlib.symmetrize(sub_A)

        '''Diagonalize the subspace Hamiltonian, and sorted.
        sub_eigenvalue[:k] are smallest k eigenvalues'''
        sub_eigenvalue, sub_eigenket = np.linalg.eigh(sub_A)
        full_guess = np.dot(V[:,:new_m], sub_eigenket[:, :k])

        '''residual = AX - XΩ = AVx - XΩ = Wx - XΩ'''
        residual = np.dot(W[:,:new_m], sub_eigenket[:,:k])
        residual -= full_guess*sub_eigenvalue[:k]

        r_norms = np.linalg.norm(residual, axis=0).tolist()
        max_norm = np.max(r_norms)
        if max_norm < tol or i == (max-1):
            break

        '''index for unconverged residuals'''
        index = [r_norms.index(i) for i in r_norms if i > tol]
        '''precondition the unconverged residuals'''
        new_guess = TDA_A_diag_preconditioner(
                        residual = residual[:,index],
                  sub_eigenvalue = sub_eigenvalue[:k][index],
                           hdiag = max_vir_hdiag)

        '''orthonormalize the new guess against basis and put into V holder'''
        m = new_m
        V, new_m = mathlib.Gram_Schmidt_fill_holder(V, m, new_guess)

    sTDA_D_end = time.time()
    sTDA_D = sTDA_D_end - sTDA_D_start
    print('sTDA A diagonalized in', i, 'steps; ', '%.4f'%sTDA_D, 'seconds' )
    print('threshold =', tol)
    print('sTDA excitation energies:')
    print(sub_eigenvalue[:k]*parameterlib.Hartree_to_eV)

    U = np.zeros((n_occ,n_vir,k))
    U[:,:max_vir,:] = full_guess.reshape(n_occ,max_vir,k)
    U = U.reshape(A_size, k)
    omega = sub_eigenvalue[:k]*parameterlib.Hartree_to_eV
    return U, omega

def sTDA_preconditioner(residual, sub_eigenvalue, tol=args.precond_TOL,
                        current_dic={}, matrix_vector_product=sTDA_mv,full_guess=None, return_index=None,
                        W_H=None, V_H=None, sub_A_H=None, max = 30):
    '''sTDA preconditioner
       (A - Ω*I)^-1 P = X
       AX - XΩ = P
       P is residuals (in big Davidson's loop) to be preconditioned
    '''
    p_start = time.time()

    '''number of vectors to be preconditioned'''
    N_vectors = residual.shape[1]
    Residuals = residual.reshape(n_occ,n_vir,-1)
    omega = sub_eigenvalue
    P = Residuals[:,:max_vir,:]
    P = P.reshape(A_reduced_size,-1)

    pnorm = np.linalg.norm(P, axis=0, keepdims = True)
    P = P/pnorm

    start = time.time()
    tol = args.precond_TOL # Convergence tolerance
       # Maximum number of iterations

    V = np.zeros((A_reduced_size, (max+1)*N_vectors))
    W = np.zeros((A_reduced_size, (max+1)*N_vectors))
    count = 0

    '''now V and W are empty holders, 0 vectors
       W = sTDA_mv(V)
       count is the amount of vectors that already sit in the holder
       in each iteration, V and W will be filled/updated with new guess basis
       which is the preconditioned residuals
    '''

    '''initial guess: DX - XΩ = P
       Dp is the preconditioner
       <t: returns np.sign(D)*t; else: D
    '''
    t = 1e-10
    Dp = np.repeat(hdiag.reshape(-1,1), N_vectors, axis=1) - omega
    Dp = np.where(abs(Dp)<t, np.sign(Dp)*t, Dp)
    Dp = Dp.reshape(n_occ, n_vir, -1)
    D = Dp[:,:max_vir,:].reshape(A_reduced_size,-1)
    inv_D = 1/D

    '''generate initial guess'''
    Xig = P*inv_D
    count = 0
    V, new_count = mathlib.Gram_Schmidt_fill_holder(V, count, Xig)

    origin_dic = current_dic.copy()
    current_dic['preconditioning'] = []
    mvcost = 0
    GScost = 0
    subcost = 0
    subgencost = 0

    for i in range(max):

        '''project sTDA_A matrix and vector P into subspace'''
        mvstart = time.time()
        W[:, count:new_count] = matrix_vector_product(V[:, count:new_count])
        mvend = time.time()
        mvcost += mvend - mvstart

        substart = time.time()
        sub_P= np.dot(V[:,:new_count].T, P)
        sub_A = np.dot(V[:,:new_count].T, W[:,:new_count])
        subend = time.time()
        subgencost += subend - substart

        sub_A = mathlib.symmetrize(sub_A)
        m = np.shape(sub_A)[0]

        substart = time.time()
        sub_guess = mathlib.solve_AX_Xla_B(sub_A, omega, sub_P)
        subend = time.time()
        subcost += subend - substart

        full_guess = np.dot(V[:,:new_count], sub_guess)
        residual = np.dot(W[:,:new_count], sub_guess) - full_guess*omega - P

        r_norms = np.linalg.norm(residual, axis=0).tolist()
        current_dic['preconditioning'].append(
                                    {'precondition residual norms': r_norms})

        max_norm = np.max(r_norms)
        if max_norm < tol or i == (max-1):
            break

        '''index of unconverged states'''
        index = [r_norms.index(i) for i in r_norms if i > tol]

        '''precondition the unconverged residuals'''
        new_guess = residual[:,index]*inv_D[:,index]


        GSstart = time.time()
        count = new_count
        V, new_count = mathlib.Gram_Schmidt_fill_holder(V, count, new_guess)
        GSend = time.time()
        GScost += GSend - GSstart

    p_end = time.time()
    p_cost = p_end - p_start

    if i == (max -1):
        print('_____sTDA Preconditioner Failed Due to Iteration Limit _______')
        print('failed after ', i, 'steps,', '%.4f'%p_cost,'s')
        print('current residual norms', r_norms)
    else:
        print('sTDA precond Done after', i, 'steps;', '%.4f'%p_cost,'seconds')

    print('max_norm = ', '%.2e'%max_norm)
    for enrty in ['subgencost', 'mvcost', 'GScost', 'subcost']:
        cost = locals()[enrty]
        print("{:<10} {:<5.4f}s  {:<5.2%}".format(enrty, cost, cost/p_cost))
    full_guess = full_guess*pnorm

    U = np.zeros((n_occ,n_vir,N_vectors))
    U[:,:max_vir,:] = full_guess.reshape(n_occ,max_vir,-1)

    if max_vir < n_vir:
        ''' DX2 - X2*Omega = P2'''
        P2 = Residuals[:,max_vir:,:]
        P2 = P2.reshape(n_occ*(n_vir-max_vir),-1)

        D2 = Dp[:,max_vir:,:]
        D2 = D2.reshape(n_occ*(n_vir-max_vir),-1)
        X2 = (P2/D2).reshape(n_occ,n_vir-max_vir,-1)
        U[:,max_vir:,:] = X2

    U = U.reshape(A_size, -1)

    '''if we want to know more about the preconditioning process,
        return the current_dic, rather than origin_dic'''
    return U, origin_dic

def Jacobi_preconditioner(residual, sub_eigenvalue, current_dic, full_guess,
                return_index = None, W_H = None, V_H = None, sub_A_H = None,
                DKapp = args.DKapp):
    '''(1-uu*)(A-Ω*I)t = -B
        (1-uu*)Kz = y
        Kz - uu*Kz = y
        Kz + αu = y
        z =  K^-1y - αK^-1u
       B is residual, we want to solve t (approximately)
       z approximates t
       z = (A-Ω*I)^(-1)*(-B) - α(A-Ω*I)^(-1)*u
        let K_inv_y = (A-Ω*I)^(-1)*(-B)
        and K_inv_u = (A-Ω*I)^(-1)*u
       z = K_inv_y - α*K_inv_u
       where α = [u*(A-Ω*I)^(-1)y]/[u*(A-Ω*I)^(-1)u]  (using uz = 0)
       first, solve (A-Ω*I)^(-1)y and (A-Ω*I)^(-1)u
    '''
    B = residual
    omega = sub_eigenvalue
    u = full_guess

    if DKapp == False:
        K_inv_y, NA_dic = sTDA_preconditioner(-B, omega)
        K_inv_u, NA_dic = sTDA_preconditioner(u, omega)
    else:
        K_inv_y = TDA_A_diag_preconditioner(
                                residual=-B, sub_eigenvalue=omega, hdiag=hdiag)
        K_inv_u = TDA_A_diag_preconditioner(
                                residual=u, sub_eigenvalue=omega, hdiag=hdiag)

    n = np.multiply(u, K_inv_y).sum(axis=0)
    d = np.multiply(u, K_inv_u).sum(axis=0)
    Alpha = n/d
    print('N in Jacobi =', np.average(n))
    print('D in Jacobi =', np.average(d))
    print('Alpha in Jacobi =', np.average(Alpha))

    z = K_inv_y -  Alpha*K_inv_u

    return z, current_dic

def on_the_fly_Hx(W, V, sub_A, x):
    def Qx(V, x):
        '''Qx = (1 - V*V.T)*x = x - V*V.T*x'''
        VX = np.dot(V.T,x)
        x -= np.dot(V,VX)
        return x
    '''on-the-fly compute H'x
       H′ ≡ W*V.T + V*W.T − V*a*V.T + Q*K*Q
       K approximates H, here K = sTDA_A
       H′ ≡ W*V.T + V*W.T − V*a*V.T + (1-V*V.T)sTDA_A(1-V*V.T)
       H′x ≡ a + b − c + d
    '''
    a = einsum('ij, jk, kl -> il', W, V.T, x)
    b = einsum('ij, jk, kl -> il', V, W.T, x)
    c = einsum('ij, jk, kl, lm -> im', V, sub_A, V.T, x)
    d = Qx(V, sTDA_mv(Qx(V, x)))
    Hx = a + b - c + d
    return Hx

def new_ES(full_guess, return_index, W_H, V_H, sub_A_H,
                        residual=None, sub_eigenvalue=None, current_dic=None):
    '''new eigenvalue solver, to diagonalize the H'
       the traditional Davidson to diagonalize the H' matrix
       W_H, V_H, sub_A_H are from the exact H
    '''
    new_ES_start = time.time()
    tol = args.eigensolver_tol
    max = 30

    k = args.nstates
    m = min([k+8, 2*k, A_size])

    V = np.zeros((A_size, max*k + m))
    W = np.zeros_like(V)

    '''sTDA as initial guess'''
    V = sTDA_eigen_solver(m, V)
    W[:,:m] = on_the_fly_Hx(W_H, V_H, sub_A_H, V[:, :m])

    for i in range(max):
        sub_A = np.dot(V[:,:m].T, W[:,:m])
        sub_eigenvalue, sub_eigenket = np.linalg.eigh(sub_A)
        residual = np.dot(W[:,:m], sub_eigenket[:,:k])
        residual -= np.dot(V[:,:m], sub_eigenket[:,:k] * sub_eigenvalue[:k])

        r_norms = np.linalg.norm(residual, axis=0).tolist()
        max_norm = np.max(r_norms)
        if max_norm < tol or i == (max-1):
            break
        index = [r_norms.index(i) for i in r_norms if i > tol]

        new_guess = TDA_A_diag_preconditioner(
                                          residual=residual[:,index],
                                    sub_eigenvalue=sub_eigenvalue[:k][index],
                                             hdiag=hdiag)
        V, new_m = mathlib.Gram_Schmidt_fill_holder(V, m, new_guess)
        W[:, m:new_m] = on_the_fly_Hx(W_H, V_H, sub_A_H, V[:, m:new_m])
        m = new_m

    full_guess = np.dot(V[:,:m], sub_eigenket[:,:k])

    new_ES_end = time.time()
    new_ES_cost = new_ES_end - new_ES_start
    print('H_app diagonalization done in',i,'steps; ','%.2f'%new_ES_cost, 's')
    print('threshold =', tol)
    return full_guess[:,return_index], current_dic

def gen_TDA_lib():
    i_lib={}
    p_lib={}
    i_lib['sTDA']   = sTDA_eigen_solver
    i_lib['Adiag']  = TDA_A_diag_initial_guess
    p_lib['sTDA']   = sTDA_preconditioner
    p_lib['Adiag']  = TDA_A_diag_preconditioner
    p_lib['Jacobi'] = Jacobi_preconditioner
    p_lib['new_ES'] = new_ES
    return i_lib, p_lib

def fill_dictionary(dic,init,prec,k,icost,pcost,wall_time,N_itr,N_mv,
            initial_energies=None,energies=None,difference=None,overlap=None,
            tensor_alpha=None, initial_tensor_alpha=None):
    dic['initial guess'] = init
    dic['preconditioner'] = prec
    dic['nstate'] = k
    dic['molecule'] = basename
    dic['method'] = args.method
    dic['functional'] = args.functional
    dic['threshold'] = args.conv_tolerance
    dic['SCF time'] = kernel_t
    dic['Initial guess time'] = icost
    dic['initial guess threshold'] = args.initial_TOL
    dic['New guess generating time'] = pcost
    dic['preconditioner threshold'] = args.precond_TOL
    dic['total time'] = wall_time
    dic['excitation energy(eV)'] = energies
    dic['iterations'] = N_itr
    dic['A matrix size'] = A_size
    dic['final subspace size'] = N_mv
    dic['ax'] = a_x
    dic['alpha'] = alpha
    dic['beta'] = beta
    dic['virtual truncation tol'] = args.truncate_virtual
    dic['n_occ'] = n_occ
    dic['n_vir'] = n_vir
    dic['max_vir'] = max_vir
    dic['semiempirical_difference'] = difference
    dic['overlap'] = overlap
    dic['initial_energies'] = initial_energies
    dic['Dynamic polarizability wavelength'] = args.dynpol_omega
    dic['Dynamic polarizability tensor alpha'] = tensor_alpha
    dic['Dynamic polarizability initial tensor alpha'] = initial_tensor_alpha
    return dic

def on_the_fly_tuning(sub_A, V):
    print('checking commutator_se norm')
    # print('{:<8s}{:<8s}{:<20s}'.format('beta','alpha','commutator_se_norm'))
    smallest_norm=1000
    opt_alpha=0
    opt_beta=0
    for try_beta in args.beta_list:
        for try_alpha in args.alpha_list:
            tensors = on_the_fly_tensors(alpha=try_alpha, beta=try_beta)
            sub_A_se = np.dot(V.T, tensors.full_sTDA_mv(V))
            commutator = np.dot(sub_A,sub_A_se) - np.dot(sub_A_se,sub_A)
            commutator_se_norm = np.linalg.norm(commutator)
            if commutator_se_norm < smallest_norm :
                smallest_norm = commutator_se_norm
                opt_alpha=try_alpha
                opt_beta=try_beta
            # print("{:<8.2f}{:<8.2f}{:<20.15f}".format(try_beta,try_alpha,commutator_se_norm))
    print('the predicted best alpha beta pairs is:')
    print("{:<8.2f}{:<8.2f}{:<20.15f}".format(opt_beta,opt_alpha,smallest_norm))
    return opt_alpha, opt_beta

def Davidson(init, prec, k=args.nstates, tol=args.conv_tolerance):
    '''Davidson frame, we can use different initial guess and preconditioner'''
    D_start = time.time()
    Davidson_dic = {}
    Davidson_dic['iteration'] = []
    iteration_list = Davidson_dic['iteration']

    TDA_i_lib, TDA_p_lib = gen_TDA_lib()
    initial_guess = TDA_i_lib[init]
    new_guess_generator = TDA_p_lib[prec]

    print('Initial guess:  ', init)
    print('Preconditioner: ', prec)

    init_start = time.time()
    max = args.max
    m = 0
    new_m = min([k + args.extrainitial, 2*k, A_size])
    V = np.zeros((A_size, max*k + new_m))
    W = np.zeros_like(V)
    V[:, :new_m], initial_energies = initial_guess(new_m)
    init_end = time.time()

    init_time = init_end - init_start
    print('initial guess time %.4f seconds'%init_time)

    Pcost = 0
    MVcost = 0
    for ii in range(max):
        print('\n')
        print('Iteration ', ii)
        istart = time.time()

        MV_start = time.time()
        W[:, m:new_m] = TDA_matrix_vector(V[:,m:new_m])
        MV_end = time.time()
        iMVcost = MV_end - MV_start
        MVcost += iMVcost
        sub_A = np.dot(V[:,:new_m].T, W[:,:new_m])
        sub_A = mathlib.symmetrize(sub_A)
        print('subspace size: ', np.shape(sub_A)[0])

        sub_eigenvalue, sub_eigenket = np.linalg.eigh(sub_A)
        full_guess = np.dot(V[:,:new_m], sub_eigenket[:,:k])
        AV = np.dot(W[:,:new_m], sub_eigenket[:,:k])
        residual = AV - full_guess * sub_eigenvalue[:k]

        r_norms = np.linalg.norm(residual, axis=0).tolist()
        max_norm = np.max(r_norms)

        iteration_list.append({})
        current_dic = iteration_list[ii]
        current_dic['residual norms'] = r_norms

        print('maximum residual norm %.2e'%max_norm)
        if max_norm < tol or ii == (max-1):
            iend = time.time()
            icost = iend - istart
            current_dic['iteration total cost'] = icost
            current_dic['iteration MV cost'] = iMVcost
            iteration_list[ii] = current_dic
            print('iMVcost %.4f'%iMVcost)
            print('icost %.4f'%icost)
            print('Davidson procedure Done \n')
            break

        index = [r_norms.index(i) for i in r_norms if i > tol]

        P_start = time.time()

        new_guess, current_dic = new_guess_generator(
                                residual = residual[:,index],
                          sub_eigenvalue = sub_eigenvalue[:k][index],
                   matrix_vector_product = sTDA_mv,
                             current_dic = current_dic,
                              full_guess = full_guess[:,index],
                            return_index = index,
                                     W_H = W[:,:m],
                                     V_H = V[:,:m],
                                 sub_A_H = sub_A)

        P_end = time.time()

        iteration_list[ii] = current_dic

        Pcost += P_end - P_start

        m = new_m
        V, new_m = mathlib.Gram_Schmidt_fill_holder(V, m, new_guess)
        print('new generated guesses:', new_m - m)

        iend = time.time()
        icost = iend - istart
        current_dic['iteration cost'] = icost
        current_dic['iteration MV cost'] = iMVcost
        iteration_list[ii] = current_dic
        print('iMVcost %.4f'%iMVcost)
        print('icost %.4f'%icost)

    energies = sub_eigenvalue[:k]*parameterlib.Hartree_to_eV
    V_basis = V[:,:new_m]
    W_basis = W[:,:new_m]

    D_end = time.time()
    Dcost = D_end - D_start

    Davidson_dic = fill_dictionary(Davidson_dic, init=init, prec=prec, k=k,
                                icost=init_time, pcost=Pcost, wall_time=Dcost,
            energies = energies.tolist(), N_itr=ii+1, N_mv=np.shape(sub_A)[0],
            initial_energies=initial_energies.tolist())
    if ii == max-1:
        print('========== Davidson Failed Due to Iteration Limit ============')
        print('current residual norms', r_norms)
    else:
        print('------- Davidson done -------')
    print('max_norm = ', max_norm)
    print('Total steps =', ii+1)
    print('Total time: %.4f seconds'%Dcost)
    print('MVcost %.4f'%MVcost)
    print('Final subspace shape = %s'%np.shape(sub_A)[0])
    print('Precond time: %.4f seconds'%Pcost, '{:.2%}'.format(Pcost/Dcost))
    return (energies, full_guess, AV, residual, Davidson_dic, V_basis, W_basis,
                                                                          sub_A)

def TDDFT_A_diag_initial_guess(V_holder, W_holder, new_m, hdiag=hdiag):
    hdiag = hdiag.reshape(-1,)
    Dsort = hdiag.argsort()
    V_holder[:,:new_m], energies = TDA_A_diag_initial_guess(new_m, hdiag=hdiag)
    return (V_holder, W_holder, new_m, energies, V_holder[:,:new_m],
                                                            W_holder[:,:new_m])

def TDDFT_A_diag_preconditioner(R_x, R_y, omega, hdiag, tol=None):
    '''preconditioners for each corresponding residual (state)'''
    hdiag = hdiag.reshape(-1,1)
    k = R_x.shape[1]
    t = 1e-14
    d = np.repeat(hdiag.reshape(-1,1), k, axis=1)

    D_x = d - omega
    D_x = np.where( abs(D_x) < t, np.sign(D_x)*t, D_x)
    D_x_inv = D_x**-1

    D_y = d + omega
    D_y = np.where( abs(D_y) < t, np.sign(D_y)*t, D_y)
    D_y_inv = D_y**-1

    X_new = R_x*D_x_inv
    Y_new = R_y*D_y_inv

    return X_new, Y_new

def sTDDFT_eigen_solver(k, tol=args.initial_TOL):
    '''[ A' B' ] X - [1   0] Y Ω = 0
       [ B' A' ] Y   [0  -1] X   = 0
    '''
    max = 30
    sTDDFT_start = time.time()
    print('setting initial guess')
    print('sTDDFT Convergence tol = %.2e'%tol)
    m = 0
    new_m = min([k+8, 2*k, A_reduced_size])
    V_holder = np.zeros((A_reduced_size, (max+1)*k))
    W_holder = np.zeros_like(V_holder)

    U1_holder = np.zeros_like(V_holder)
    U2_holder = np.zeros_like(V_holder)

    '''set up initial guess V W, transformed vectors U1 U2'''
    V_holder, W_holder, new_m, energies, Xig, Yig = TDDFT_A_diag_initial_guess(
        V_holder=V_holder, W_holder=W_holder, new_m=new_m, hdiag=max_vir_hdiag)

    subcost = 0
    Pcost = 0
    MVcost = 0
    GScost = 0
    subgencost = 0
    for ii in range(max):
        V = V_holder[:,:new_m]
        W = W_holder[:,:new_m]
        '''U1 = AV + BW
           U2 = AW + BV'''

        MV_start = time.time()
        U1_holder[:, m:new_m], U2_holder[:, m:new_m] = sTDDFT_mv(
                                            X=V[:, m:new_m], Y=W[:, m:new_m])
        MV_end = time.time()
        MVcost += MV_end - MV_start

        U1 = U1_holder[:,:new_m]
        U2 = U2_holder[:,:new_m]

        subgenstart = time.time()
        a = np.dot(V.T, U1)
        a += np.dot(W.T, U2)

        b = np.dot(V.T, U2)
        b += np.dot(W.T, U1)

        sigma = np.dot(V.T, V)
        sigma -= np.dot(W.T, W)

        pi = np.dot(V.T, W)
        pi -= np.dot(W.T, V)


        a = mathlib.symmetrize(a)
        b = mathlib.symmetrize(b)
        sigma = mathlib.symmetrize(sigma)
        pi = mathlib.anti_symmetrize(pi)

        subgenend = time.time()
        subgencost += subgenend - subgenstart

        '''solve the eigenvalue omega in the subspace'''
        subcost_start = time.time()
        omega, x, y = mathlib.TDDFT_subspace_eigen_solver(a, b, sigma, pi, k)
        subcost_end = time.time()
        subcost += subcost_end - subcost_start

        '''compute the residual
           R_x = U1x + U2y - X_full*omega
           R_y = U2x + U1y + Y_full*omega
           X_full = Vx + Wy
           Y_full = Wx + Vy
        '''

        X_full = np.dot(V,x)
        X_full += np.dot(W,y)

        Y_full = np.dot(W,x)
        Y_full += np.dot(V,y)

        R_x = np.dot(U1,x)
        R_x += np.dot(U2,y)
        R_x -= X_full*omega

        R_y = np.dot(U2,x)
        R_y += np.dot(U1,y)
        R_y += Y_full*omega

        residual = np.vstack((R_x, R_y))
        r_norms = np.linalg.norm(residual, axis=0).tolist()
        max_norm = np.max(r_norms)
        if max_norm < tol or ii == (max -1):
            break

        index = [r_norms.index(i) for i in r_norms if i > tol]

        '''preconditioning step'''
        X_new, Y_new = TDDFT_A_diag_preconditioner(R_x=R_x[:,index],
                    R_y=R_y[:,index], omega=omega[index], hdiag = max_vir_hdiag)

        '''GS and symmetric orthonormalization'''
        m = new_m
        GScost_start = time.time()
        V_holder, W_holder, new_m = mathlib.VW_Gram_Schmidt_fill_holder\
                                        (V_holder, W_holder, m, X_new, Y_new)
        GScost_end = time.time()
        GScost += GScost_end - GScost_start

        if new_m == m:
            print('All new guesses kicked out during GS orthonormalization')
            break

    sTDDFT_end = time.time()

    sTDDFT_cost = sTDDFT_end - sTDDFT_start

    if ii == (max -1):
        print('========= sTDDFT Failed Due to Iteration Limit=================')
        print('sTDDFT diagonalization failed')
        print('current residual norms', r_norms)
    else:
        print('sTDDFT diagonalization Converged' )

    print('after ', ii+1, 'iterations; %.4f'%sTDDFT_cost, 'seconds')
    print('final subspace', sigma.shape[0])
    print('max_norm = ', '%.2e'%max_norm)
    for enrty in ['MVcost','GScost','subgencost','subcost']:
        cost = locals()[enrty]
        print("{:<10} {:<5.4f}s {:<5.2%}".format(enrty, cost, cost/sTDDFT_cost))
    X = np.zeros((n_occ,n_vir,k))
    Y = np.zeros((n_occ,n_vir,k))

    X[:,:max_vir,:] = X_full.reshape(n_occ,max_vir,-1)
    Y[:,:max_vir,:] = Y_full.reshape(n_occ,max_vir,-1)

    X = X.reshape(A_size, -1)
    Y = Y.reshape(A_size, -1)

    energies = omega*parameterlib.Hartree_to_eV
    print('sTDDFT excitation energy:')
    print(energies)
    return energies, X, Y

def sTDDFT_initial_guess(V_holder, W_holder, new_m):
    energies, X_new_backup, Y_new_backup = sTDDFT_eigen_solver(new_m)
    V_holder, W_holder, new_m = mathlib.VW_Gram_Schmidt_fill_holder(
                            V_holder, W_holder, 0,  X_new_backup, Y_new_backup)
    return V_holder, W_holder, new_m, energies, X_new_backup, Y_new_backup

def sTDDFT_preconditioner(Rx, Ry, omega, tol=args.precond_TOL):
    ''' [ A' B' ] - [1  0]X  Ω = P'''
    ''' [ B' A' ]   [0 -1]Y    = Q'''
    ''' P = Rx '''
    ''' Q = Ry '''

    print('sTDDFT_preconditioner conv', tol)
    max = 30
    sTDDFT_start = time.time()
    k = len(omega)
    m = 0

    Rx = Rx.reshape(n_occ,n_vir,-1)
    Ry = Ry.reshape(n_occ,n_vir,-1)

    P = Rx[:,:max_vir,:].reshape(A_reduced_size,-1)
    Q = Ry[:,:max_vir,:].reshape(A_reduced_size,-1)

    initial_start = time.time()
    V_holder = np.zeros((A_reduced_size, (max+1)*k))
    W_holder = np.zeros_like(V_holder)

    U1_holder = np.zeros_like(V_holder)
    U2_holder = np.zeros_like(V_holder)

    '''normalzie the RHS'''
    PQ = np.vstack((P,Q))
    pqnorm = np.linalg.norm(PQ, axis=0, keepdims = True)

    P /= pqnorm
    Q /= pqnorm

    X_new, Y_new  = TDDFT_A_diag_preconditioner(R_x=P, R_y=Q, omega=omega,
                                                            hdiag=max_vir_hdiag)
    V_holder, W_holder, new_m = mathlib.VW_Gram_Schmidt_fill_holder(
                                    V_holder, W_holder, 0,  X_new, Y_new)
    initial_end = time.time()
    initial_cost = initial_end - initial_start

    subcost = 0
    Pcost = 0
    MVcost = 0
    GScost = 0
    subgencost = 0
    for ii in range(max):
        V = V_holder[:,:new_m]
        W = W_holder[:,:new_m]

        '''U1 = AV + BW
           U2 = AW + BV'''

        MV_start = time.time()
        U1_holder[:, m:new_m], U2_holder[:, m:new_m] = sTDDFT_mv(
                                            X=V[:, m:new_m], Y=W[:, m:new_m])
        MV_end = time.time()
        MVcost += MV_end - MV_start

        U1 = U1_holder[:,:new_m]
        U2 = U2_holder[:,:new_m]

        subgenstart = time.time()
        a = np.dot(V.T, U1)
        a += np.dot(W.T, U2)

        b = np.dot(V.T, U2)
        b += np.dot(W.T, U1)

        sigma = np.dot(V.T, V)
        sigma -= np.dot(W.T, W)

        pi = np.dot(V.T, W)
        pi -= np.dot(W.T, V)

        '''p = VP + WQ
           q = WP + VQ'''
        p = np.dot(V.T, P)
        p += np.dot(W.T, Q)

        q = np.dot(W.T, P)
        q += np.dot(V.T, Q)

        subgenend = time.time()
        subgencost += subgenend - subgenstart

        a = mathlib.symmetrize(a)
        b = mathlib.symmetrize(b)
        sigma = mathlib.symmetrize(sigma)
        pi = mathlib.anti_symmetrize(pi)

        '''solve the x & y in the subspace'''
        subcost_start = time.time()
        x, y = mathlib.TDDFT_subspace_liear_solver(a, b, sigma, pi, p, q, omega)
        subcost_end = time.time()
        subcost += subcost_end - subcost_start

        '''compute the residual
           R_x = U1x + U2y - X_full*omega - P
           R_y = U2x + U1y + Y_full*omega - Q
           X_full = Vx + Wy
           Y_full = Wx + Vy
        '''

        X_full = np.dot(V,x)
        X_full += np.dot(W,y)

        Y_full = np.dot(W,x)
        Y_full += np.dot(V,y)

        R_x = np.dot(U1,x)
        R_x += np.dot(U2,y)
        R_x -= X_full*omega
        R_x -= P

        R_y = np.dot(U2,x)
        R_y += np.dot(U1,y)
        R_y += Y_full*omega
        R_y -= Q

        residual = np.vstack((R_x,R_y))
        r_norms = np.linalg.norm(residual, axis=0).tolist()
        max_norm = np.max(r_norms)
        if max_norm < tol or ii == (max -1):
            break
        index = [r_norms.index(i) for i in r_norms if i > tol]

        '''preconditioning step'''
        Pstart = time.time()
        X_new, Y_new = TDDFT_A_diag_preconditioner(R_x[:,index], R_y[:,index],
                                            omega[index], hdiag = max_vir_hdiag)
        Pend = time.time()
        Pcost += Pend - Pstart

        '''GS and symmetric orthonormalization'''
        m = new_m
        GS_start = time.time()
        V_holder, W_holder, new_m = mathlib.VW_Gram_Schmidt_fill_holder(\
                                            V_holder, W_holder, m, X_new, Y_new)
        GS_end = time.time()
        GScost += GS_end - GS_start

        if new_m == m:
            print('All new guesses kicked out during GS orthonormalization')
            break

    sTDDFT_end = time.time()

    P_cost = sTDDFT_end - sTDDFT_start

    if ii == (max -1):
        print('========== sTDDFT_precond Failed Due to Iteration Limit========')
        print('sTDDFT preconditioning failed')
        print('current residual norms', r_norms)
    else:
        print('sTDDFT preconditioning Done')
    print('after',ii+1,'steps; %.4f'%P_cost,'s')
    print('final subspace', sigma.shape[0])
    print('max_norm = ', '%.2e'%max_norm)
    for enrty in ['initial_cost','MVcost','GScost','subgencost','subcost']:
        cost = locals()[enrty]
        print("{:<10} {:<5.4f}s  {:<5.2%}".format(enrty, cost, cost/P_cost))

    X_full = X_full*pqnorm
    Y_full = Y_full*pqnorm

    X = np.zeros((n_occ,n_vir,k))
    Y = np.zeros((n_occ,n_vir,k))

    X[:,:max_vir,:] = X_full.reshape(n_occ,max_vir,k)
    Y[:,:max_vir,:] = Y_full.reshape(n_occ,max_vir,k)

    if max_vir < n_vir:
        P2 = Rx[:,max_vir:,:].reshape(n_occ*(n_vir-max_vir),-1)
        Q2 = Ry[:,max_vir:,:].reshape(n_occ*(n_vir-max_vir),-1)

        X2, Y2 = TDDFT_A_diag_preconditioner(R_x=P2, R_y=Q2, omega=omega,
                                                hdiag=delta_hdiag[:,max_vir:])
        X[:,max_vir:,:] = X2.reshape(n_occ,n_vir-max_vir,-1)
        Y[:,max_vir:,:] = Y2.reshape(n_occ,n_vir-max_vir,-1)

    X = X.reshape(A_size,-1)
    Y = Y.reshape(A_size,-1)

    return X, Y

def gen_TDDFT_lib():
    i_lib={}
    p_lib={}
    i_lib['sTDDFT'] = sTDDFT_initial_guess
    i_lib['Adiag']  = TDDFT_A_diag_initial_guess
    p_lib['sTDDFT'] = sTDDFT_preconditioner
    p_lib['Adiag']  = TDDFT_A_diag_preconditioner
    return i_lib, p_lib

def TDDFT_eigen_solver(init, prec, k=args.nstates, tol=args.conv_tolerance):
    '''[ A B ] X - [1   0] Y Ω = 0
       [ B A ] Y   [0  -1] X   = 0
    '''
    Davidson_dic = {}
    Davidson_dic['iteration'] = []
    iteration_list = Davidson_dic['iteration']

    print('Initial guess:  ', init)
    print('Preconditioner: ', prec)
    print('A matrix size = ', A_size)

    TDDFT_start = time.time()
    max = args.max
    m = 0

    new_m = min([k + args.extrainitial, 2*k, A_size])

    TDDFT_i_lib, TDDFT_p_lib = gen_TDDFT_lib()

    initial_guess = TDDFT_i_lib[init]
    new_guess_generator = TDDFT_p_lib[prec]

    V_holder = np.zeros((A_size, (max+3)*k))
    W_holder = np.zeros_like(V_holder)

    U1_holder = np.zeros_like(V_holder)
    U2_holder = np.zeros_like(V_holder)

    init_start = time.time()
    V_holder, W_holder, new_m, initial_energies, X_ig, Y_ig =\
                                    initial_guess(V_holder, W_holder, new_m)
    init_end = time.time()
    init_time = init_end - init_start

    initial_energies = initial_energies.tolist()[:k]

    print('new_m =', new_m)
    print('initial guess done')

    Pcost = 0
    for ii in range(max):
        print('\niteration', ii)
        show_memory_info('beginning of step '+ str(ii))

        V = V_holder[:,:new_m]
        W = W_holder[:,:new_m]
        '''U1 = AV + BW
           U2 = AW + BV'''
        U1_holder[:, m:new_m], U2_holder[:, m:new_m] =\
                            TDDFT_matrix_vector(V[:, m:new_m], W[:, m:new_m])

        U1 = U1_holder[:,:new_m]
        U2 = U2_holder[:,:new_m]

        a = np.dot(V.T, U1)
        a += np.dot(W.T, U2)

        b = np.dot(V.T, U2)
        b += np.dot(W.T, U1)

        sigma = np.dot(V.T, V)
        sigma -= np.dot(W.T, W)

        pi = np.dot(V.T, W)
        pi -= np.dot(W.T, V)

        a = mathlib.symmetrize(a)
        b = mathlib.symmetrize(b)
        sigma = mathlib.symmetrize(sigma)
        pi = mathlib.anti_symmetrize(pi)

        print('subspace size: %s' %sigma.shape[0])

        omega, x, y = mathlib.TDDFT_subspace_eigen_solver(a, b, sigma, pi, k)

        '''compute the residual
           R_x = U1x + U2y - X_full*omega
           R_y = U2x + U1y + Y_full*omega
           X_full = Vx + Wy
           Y_full = Wx + Vy
        '''

        X_full = np.dot(V,x)
        X_full += np.dot(W,y)

        Y_full = np.dot(W,x)
        Y_full += np.dot(V,y)

        R_x = np.dot(U1,x)
        R_x += np.dot(U2,y)
        R_x -= X_full*omega

        R_y = np.dot(U2,x)
        R_y += np.dot(U1,y)
        R_y += Y_full*omega

        residual = np.vstack((R_x, R_y))
        r_norms = np.linalg.norm(residual, axis=0).tolist()

        iteration_list.append({})
        current_dic = iteration_list[ii]
        current_dic['residual norms'] = r_norms
        iteration_list[ii] = current_dic

        max_norm = np.max(r_norms)
        print('Maximum residual norm: ', '%.2e'%max_norm)
        if max_norm < tol or ii == (max -1):
            print('TDDFT precedure Done\n')
            break
        index = [r_norms.index(i) for i in r_norms if i > tol]
        index = [i for i,R in enumerate(r_norms) if R > tol]
        print('unconverged states', index)

        P_start = time.time()
        X_new, Y_new = new_guess_generator(\
                            R_x[:,index], R_y[:,index], omega[index])
        P_end = time.time()
        Pcost += P_end - P_start

        m = new_m
        V_holder, W_holder, new_m = mathlib.VW_Gram_Schmidt_fill_holder(\
                                        V_holder, W_holder, m, X_new, Y_new)
        print('m & new_m', m, new_m)
        if new_m == m:
            print('All new guesses kicked out during GS orthonormalization')
            break

    omega = omega*parameterlib.Hartree_to_eV

    difference = np.mean((np.array(initial_energies) - np.array(omega))**2)
    difference = float(difference)

    overlap = float(np.linalg.norm(np.dot(X_ig.T, X_full)) \
                    + np.linalg.norm(np.dot(Y_ig.T, Y_full)))

    TDDFT_end = time.time()
    TDDFT_cost = TDDFT_end - TDDFT_start

    Davidson_dic = fill_dictionary(Davidson_dic, init=init, prec=prec, k=k,
            icost=init_time, pcost=Pcost, wall_time=TDDFT_cost,
            energies=omega.tolist(), N_itr=ii+1, N_mv=np.shape(sigma)[0],
            initial_energies=initial_energies, difference=difference,
            overlap=overlap)
    if ii == (max -1):
        print('===== TDDFT Failed Due to Iteration Limit============')
        print('current residual norms', r_norms)
        print('max_norm = ', np.max(r_norms))
    else:
        print('============= TDDFT Calculation Done ==============')

    print('after', ii+1,'iterations','%.2f'%TDDFT_cost,'s')
    print('Final subspace ', sigma.shape[0])
    print('preconditioning cost', '%.4f'%Pcost, '%.2f'%(Pcost/TDDFT_cost),"%")
    print('max_norm = ', '%.2e'%max_norm)

    show_memory_info('Total TDDFT')
    return omega, X_full, Y_full, Davidson_dic

def gen_dynpol_lib():
    i_lib={}
    p_lib={}
    i_lib['sTDDFT'] = sTDDFT_preconditioner
    i_lib['Adiag']  = TDDFT_A_diag_preconditioner
    p_lib['sTDDFT'] = sTDDFT_preconditioner
    p_lib['Adiag']  = TDDFT_A_diag_preconditioner
    return i_lib, p_lib

def gen_P():
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    occidx = mo_occ > 0
    orbo = mo_coeff[:, occidx]
    orbv = mo_coeff[:,~occidx]
    int_r= mol.intor_symmetric('int1e_r')
    P = lib.einsum("xpq,pi,qa->iax", int_r, orbo, orbv.conj())
    return P

def dynamic_polarizability(init, prec):
    ''' [ A B ] - [1  0]X  w = -P'''
    ''' [ B A ]   [0 -1]Y    = -Q'''
    dp_start = time.time()

    dynpol_i_lib, dynpol_p_lib = gen_dynpol_lib()
    initial_guess = dynpol_i_lib[init]
    new_guess_generator = dynpol_p_lib[prec]

    print('Initial guess:  ', init)
    print('Preconditioner: ', prec)
    print('A matrix size = ', A_size)

    Davidson_dic = {}
    Davidson_dic['iteration'] = []
    iteration_list = Davidson_dic['iteration']

    k = len(args.dynpol_omega)
    omega =  np.zeros([3*k])
    for jj in range(k):
        '''if have 3 ω, [ω1 ω1 ω1, ω2 ω2 ω2, ω3 ω3 ω3]
           convert nm to Hartree'''
        omega[3*jj:3*(jj+1)] = 45.56337117/args.dynpol_omega[jj]

    P = gen_P()
    P = P.reshape(-1,3)

    P_origin = np.zeros_like(P)
    Q = np.zeros_like(P)

    P_origin[:,:] = P[:,:]
    Q[:,:] = P[:,:]

    pnorm = np.linalg.norm(P, axis=0, keepdims = True)
    pqnorm = pnorm * (2**0.5)
    print('pqnorm', pqnorm)
    P /= pqnorm

    P = np.tile(P,k)

    max = args.max
    tol = args.conv_tolerance
    m = 0
    V_holder = np.zeros((A_size, (max+1)*k*3))
    W_holder = np.zeros_like(V_holder)

    U1_holder = np.zeros_like(V_holder)
    U2_holder = np.zeros_like(V_holder)

    init_start = time.time()
    X_ig, Y_ig = initial_guess(-P, -Q, omega, tol=args.initial_TOL)

    alpha_omega_ig = []
    X_p_Y = X_ig + Y_ig
    X_p_Y = X_p_Y*np.tile(pqnorm,k)
    for jj in range(k):
        '''*-1 from the definition of dipole moment. *2 for double occupancy'''
        X_p_Y_tmp = X_p_Y[:,3*jj:3*(jj+1)]
        alpha_omega_ig.append(np.dot(P_origin.T, X_p_Y_tmp)*-2)
    print('initial guess of tensor alpha')
    for i in range(k):
        print(args.dynpol_omega[i],'nm')
        print(alpha_omega_ig[i])

    V_holder, W_holder, new_m = mathlib.VW_Gram_Schmidt_fill_holder(\
                                            V_holder, W_holder, 0, X_ig, Y_ig)
    init_end = time.time()
    initial_cost = init_end - init_start
    subcost = 0
    Pcost = 0
    MVcost = 0
    GScost = 0
    subgencost = 0
    for ii in range(max):
        print('Iteration', ii)

        V = V_holder[:,:new_m]
        W = W_holder[:,:new_m]

        MV_start = time.time()
        U1_holder[:, m:new_m], U2_holder[:, m:new_m] = TDDFT_matrix_vector(\
                                                V[:, m:new_m], W[:, m:new_m])
        MV_end = time.time()
        MVcost += MV_end - MV_start

        U1 = U1_holder[:,:new_m]
        U2 = U2_holder[:,:new_m]

        subgenstart = time.time()
        a = np.dot(V.T, U1)
        a += np.dot(W.T, U2)

        b = np.dot(V.T, U2)
        b += np.dot(W.T, U1)

        sigma = np.dot(V.T, V)
        sigma -= np.dot(W.T, W)

        pi = np.dot(V.T, W)
        pi -= np.dot(W.T, V)

        '''p = VP + WQ
           q = WP + VQ'''
        p = np.dot(V.T, P)
        p += np.dot(W.T, Q)

        q = np.dot(W.T, P)
        q += np.dot(V.T, Q)

        subgenend = time.time()
        subgencost += subgenend - subgenstart

        print('sigma.shape', sigma.shape)

        subcost_start = time.time()
        x, y = mathlib.TDDFT_subspace_liear_solver(a, b, sigma, pi, -p, -q, omega)
        subcost_end = time.time()
        subcost += subcost_end - subcost_start

        '''compute the residual
           R_x = U1x + U2y - X_full*omega + P
           R_y = U2x + U1y + Y_full*omega + Q
           X_full = Vx + Wy
           Y_full = Wx + Vy
        '''
        X_full = np.dot(V,x)
        X_full += np.dot(W,y)

        Y_full = np.dot(W,x)
        Y_full += np.dot(V,y)

        R_x = np.dot(U1,x)
        R_x += np.dot(U2,y)
        R_x -= X_full*omega
        R_x += P

        R_y = np.dot(U2,x)
        R_y += np.dot(U1,y)
        R_y += Y_full*omega
        R_y += Q

        residual = np.vstack((R_x,R_y))
        r_norms = np.linalg.norm(residual, axis=0).tolist()
        print('maximum residual norm: ', '%.3e'%np.max(r_norms))

        iteration_list.append({})
        current_dic = iteration_list[ii]
        current_dic['residual norms'] = r_norms
        iteration_list[ii] = current_dic

        if np.max(r_norms) < tol or ii == (max -1):
            break
        index = [r_norms.index(i) for i in r_norms if i > tol]

        Pstart = time.time()
        X_new, Y_new = new_guess_generator(R_x[:,index],
                            R_y[:,index], omega[index], tol=args.precond_TOL)
        Pend = time.time()
        Pcost += Pend - Pstart

        m = new_m
        GS_start = time.time()
        V_holder, W_holder, new_m = mathlib.VW_Gram_Schmidt_fill_holder(\
                                        V_holder, W_holder, m, X_new, Y_new)
        GS_end = time.time()
        GScost += GS_end - GS_start

        if new_m == m:
            print('All new guesses kicked out during GS orthonormalization')
            break

    dp_end = time.time()
    dp_cost = dp_end - dp_start

    if ii == (max -1):
        print('======= Dynamic polarizability Failed Due to Iteration Limit=====')
        print('dynamic polarizability failed after ', ii+1, 'iterations  ', round(dp_cost, 4), 'seconds')
        print('current residual norms', r_norms)
        print('max_norm = ', np.max(r_norms))
    else:
        print('Dynamic polarizability Converged after ', ii+1, 'iterations  ', round(dp_cost, 4), 'seconds')
        print('initial_cost', round(initial_cost,4), round(initial_cost/dp_cost * 100,2),'%')
        print('Pcost', round(Pcost,4), round(Pcost/dp_cost * 100,2),'%')
        print('MVcost', round(MVcost,4), round(MVcost/dp_cost * 100,2),'%')
        print('GScost', round(GScost,4), round(GScost/dp_cost * 100,2),'%')
        print('subcost', round(subcost,4), round(subcost/dp_cost * 100,2),'%')
        print('subgencost', round(subgencost,4), round(subgencost/dp_cost * 100,2),'%')

    print('Wavelength we look at', args.dynpol_omega)
    alpha_omega = []

    overlap = float(np.linalg.norm(np.dot(X_ig.T, X_full))\
                    + np.linalg.norm(np.dot(Y_ig.T, Y_full)))

    X_p_Y = X_full + Y_full

    X_p_Y = X_p_Y*np.tile(pqnorm,k)

    for jj in range(k):
        X_p_Y_tmp = X_p_Y[:,3*jj:3*(jj+1)]
        alpha_omega.append(np.dot(P_origin.T, X_p_Y_tmp)*-2)

    difference = 0
    for i in range(k):
        difference += np.mean((alpha_omega_ig[i] - alpha_omega[i])**2)

    difference = float(difference)

    show_memory_info('Total Dynamic polarizability')
    Davidson_dic = fill_dictionary(Davidson_dic, init=init, prec=prec, k=3*k,
            icost=initial_cost, pcost=Pcost, wall_time=dp_cost,
            energies=omega.tolist(), N_itr=ii+1, N_mv=np.shape(sigma)[0],
            difference=difference, overlap=overlap,
            tensor_alpha=[i.tolist() for i in alpha_omega],
            initial_tensor_alpha=[i.tolist() for i in alpha_omega_ig])
    return alpha_omega, Davidson_dic

def stapol_A_diag_initprec(P, hdiag=hdiag, tol=None):
    d = hdiag.reshape(-1,1)
    P = -P/d
    # P /= -d
    return P

def stapol_sTDDFT_initprec(Pr, tol=args.initial_TOL, matrix_vector_product=sTDDFT_stapol_mv):
    '''(A* + B*)X = -P
       note the negative sign of P!
       residual = (A* + B*)X + P
       X_ig = -P/d
       X_new = residual/D
    '''
    ssp_start = time.time()
    max = 30
    m = 0
    npvec = Pr.shape[1]

    P = Pr.reshape(n_occ,n_vir,-1)[:,:max_vir,:]
    P = P.reshape(A_reduced_size,-1)
    pnorm = np.linalg.norm(P, axis=0, keepdims = True)
    P /= pnorm

    V_holder = np.zeros((A_reduced_size, (max+1)*npvec))
    U_holder = np.zeros_like(V_holder)

    '''setting up initial guess'''
    init_start = time.time()
    X_ig = stapol_A_diag_initprec(P, hdiag=max_vir_hdiag)
    V_holder, new_m = mathlib.Gram_Schmidt_fill_holder(V_holder, m, X_ig)
    init_end = time.time()
    initial_cost = init_end - init_start

    subcost = 0
    Pcost = 0
    MVcost = 0
    GScost = 0
    subgencost = 0
    for ii in range(max):
        '''creating the subspace'''
        MV_start = time.time()
        '''U = AX + BX = (A+B)X'''
        U_holder[:, m:new_m] = matrix_vector_product(V_holder[:,m:new_m])
        MV_end = time.time()
        MVcost += MV_end - MV_start

        V = V_holder[:,:new_m]
        U = U_holder[:,:new_m]

        subgenstart = time.time()
        p = np.dot(V.T, P)
        a_p_b = np.dot(V.T,U)
        a_p_b = mathlib.symmetrize(a_p_b)

        subgenend = time.time()
        subgencost += subgenend - subgenstart

        '''solve the x in the subspace'''
        subcost_start = time.time()
        x = np.linalg.solve(a_p_b, -p)
        subcost_end = time.time()
        subcost += subcost_end - subcost_start

        '''compute the residual
           R = Ux + P'''
        Ux = np.dot(U,x)
        residual = Ux + P

        r_norms = np.linalg.norm(residual, axis=0).tolist()
        index = [r_norms.index(i) for i in r_norms if i > tol]
        if np.max(r_norms) < tol or ii == (max -1):
            print('Static polarizability procedure aborted')
            break

        Pstart = time.time()
        X_new = stapol_A_diag_initprec(-residual[:,index], hdiag=max_vir_hdiag)
        Pend = time.time()
        Pcost += Pend - Pstart

        '''GS and symmetric orthonormalization'''
        m = new_m
        GS_start = time.time()
        V_holder, new_m = mathlib.Gram_Schmidt_fill_holder(V_holder, m, X_new)
        GS_end = time.time()
        GScost += GS_end - GS_start
        if new_m == m:
            print('All new guesses kicked out during GS orthonormalization')
            break
    X_full = np.dot(V,x)
    '''alpha = np.dot(X_full.T, P)*-4'''

    ssp_end = time.time()
    ssp_cost = ssp_end - ssp_start

    if ii == (max -1):
        print('== sTDDFT Stapol precond Failed Due to Iteration Limit======')
        print('current residual norms', r_norms)
    else:
        print('sTDDFT Stapol precond Converged' )
    print('after', ii+1, 'steps', '%.4f'%ssp_cost,'s')
    print('conv threhsold = %.2e'%tol)
    print('final subspace:', a_p_b.shape[0])
    print('max_norm = ', '%.2e'%np.max(r_norms))
    for enrty in ['initial_cost','MVcost','GScost','subgencost','subcost']:
        cost = locals()[enrty]
        print("{:<10} {:<5.4f}s  {:<5.2%}".format(enrty, cost, cost/ssp_cost))

    X_full = X_full*pnorm

    U = np.zeros((n_occ,n_vir,npvec))
    U[:,:max_vir,:] = X_full.reshape(n_occ,max_vir,-1)[:,:,:]

    if max_vir < n_vir:
        ''' DX2 = -P2'''
        P2 = Pr.reshape(n_occ,n_vir,-1)[:,max_vir:,:]
        P2 = P2.reshape(n_occ*(n_vir-max_vir),-1)
        D2 = delta_hdiag[:,max_vir:]
        D2 = D2.reshape(n_occ*(n_vir-max_vir),-1)
        X2 = (-P2/D2).reshape(n_occ,n_vir-max_vir,-1)
        U[:,max_vir:,:] = X2
    U = U.reshape(A_size, npvec)
    return U

def gen_stapol_lib():
    i_lib={}
    p_lib={}
    i_lib['sTDDFT'] = stapol_sTDDFT_initprec
    i_lib['Adiag']  = stapol_A_diag_initprec
    p_lib['sTDDFT'] = stapol_sTDDFT_initprec
    p_lib['Adiag']  = stapol_A_diag_initprec
    return i_lib, p_lib

def static_polarizability(init, prec):
    '''(A+B)X = -P
       residual = (A+B)X + P
    '''
    print('initial guess', init)
    print('preconditioner', prec)
    sp_start = time.time()

    P = gen_P()
    P = P.reshape(-1,3)

    P_origin = np.zeros_like(P)
    P_origin[:,:] = P[:,:]

    pnorm = np.linalg.norm(P, axis=0, keepdims = True)
    P /= pnorm

    Davidson_dic = {}
    Davidson_dic['iteration'] = []
    iteration_list = Davidson_dic['iteration']

    stapol_i_lib, stapol_p_lib = gen_stapol_lib()
    initial_guess = stapol_i_lib[init]
    new_guess_generator = stapol_p_lib[prec]

    max = args.max
    tol = args.conv_tolerance
    m = 0

    V_holder = np.zeros((A_size, (max+1)*3))
    U_holder = np.zeros_like(V_holder)

    init_start = time.time()
    X_ig = initial_guess(P, tol=args.initial_TOL)

    alpha_init = np.dot((X_ig*pnorm).T, P_origin)*-4
    print('alpha tensor of initial guess:')
    print(alpha_init)

    V_holder, new_m = mathlib.Gram_Schmidt_fill_holder(V_holder, 0, X_ig)
    print('new_m =', new_m)
    init_end = time.time()
    initial_cost = init_end - init_start

    Pcost = 0
    MVcost = 0
    GScost = 0
    subgencost = 0
    for ii in range(max):
        print('\nIteration', ii)
        MV_start = time.time()
        U_holder[:, m:new_m] = \
                    static_polarizability_matrix_vector(V_holder[:,m:new_m])
        MV_end = time.time()
        MVcost += MV_end - MV_start

        V = V_holder[:,:new_m]
        U = U_holder[:,:new_m]

        subgenstart = time.time()
        p = np.dot(V.T, P)
        a_p_b = np.dot(V.T,U)
        a_p_b = mathlib.symmetrize(a_p_b)
        subgenend = time.time()

        '''solve the x in the subspace'''
        x = np.linalg.solve(a_p_b, -p)

        '''compute the residual
           R = Ux + P'''
        Ux = np.dot(U,x)
        residual = Ux + P

        r_norms = np.linalg.norm(residual, axis=0).tolist()

        iteration_list.append({})
        current_dic = iteration_list[ii]
        current_dic['residual norms'] = r_norms
        iteration_list[ii] = current_dic

        '''index for unconverged residuals'''
        index = [r_norms.index(i) for i in r_norms if i > tol]
        max_norm = np.max(r_norms)
        print('max_norm = %.2e'%max_norm)
        if max_norm < tol or ii == (max -1):
            # print('static polarizability precodure aborted\n')
            break

        '''preconditioning step'''
        Pstart = time.time()

        X_new = new_guess_generator(-residual[:,index], tol=args.precond_TOL)
        Pend = time.time()
        Pcost += Pend - Pstart

        '''GS and symmetric orthonormalization'''
        m = new_m
        GS_start = time.time()
        V_holder, new_m = mathlib.Gram_Schmidt_fill_holder(V_holder, m, X_new)
        GS_end = time.time()
        GScost += GS_end - GS_start
        if new_m == m:
            print('All new guesses kicked out during GS orthonormalization')
            break

    X_full = np.dot(V,x)
    overlap = float(np.linalg.norm(np.dot(X_ig.T, X_full)))

    X_full = X_full*pnorm

    tensor_alpha = np.dot(X_full.T, P_origin)*-4
    sp_end = time.time()
    sp_cost = sp_end - sp_start

    if ii == (max -1):
        print('==== Static polarizability Failed Due to Iteration Limit ======')
        print('current residual norms', r_norms)
        print('max_norm = ', max_norm)
    else:
        print('Static polarizability Converged')

    print('after', ii+1, 'steps; %.4f'%sp_cost,'s')
    print('final subspace', a_p_b.shape)
    print('max_norm = ', '%.2e'%np.max(r_norms))
    for enrty in ['initial_cost','MVcost','Pcost']:
        cost = locals()[enrty]
        print("{:<10} {:<5.4f}s  {:<5.2%}".format(enrty, cost, cost/sp_cost))

    difference = np.mean((alpha_init - tensor_alpha)**2)
    difference = float(difference)

    sp_end = time.time()
    spcost = sp_end - sp_start
    Davidson_dic = fill_dictionary(Davidson_dic, init=init, prec=prec, k=3,
            icost=initial_cost, pcost=Pcost, wall_time=sp_cost,
            N_itr=ii+1, N_mv=np.shape(a_p_b)[0], difference=difference,
            overlap=overlap, tensor_alpha=[i.tolist() for i in tensor_alpha],
            initial_tensor_alpha=[i.tolist() for i in alpha_init])
    return tensor_alpha, Davidson_dic

def gen_calc():
    name_dic={}
    name_dic['TDA'] = args.TDA
    name_dic['TDDFT'] = args.TDDFT
    name_dic['dynpol'] = args.dynpol
    name_dic['stapol'] = args.stapol
    name_dic['sTDA'] = args.sTDA
    name_dic['sTDDFT'] = args.sTDDFT
    name_dic['Truncate_test'] = args.Truncate_test
    name_dic['PySCF_TDDFT'] = args.pytd
    for calc in ['TDA','TDDFT','dynpol','stapol',
                        'sTDA','sTDDFT','Truncate_test','PySCF_TDDFT']:
        if name_dic[calc] == True:
            print(calc)
            return calc

def dump_yaml(Davidson_dic, calc, init, prec):
    curpath = os.getcwd()
    yamlpath = os.path.join(\
                   curpath,basename+'_'+calc+'_i_'+init+'_p_'+prec+'.yaml')
    with open(yamlpath, "w", encoding="utf-8") as f:
        yaml.dump(Davidson_dic, f)

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout




def gen_metrics(beta, alpha, ab_initio_V, ab_initio_E,
                    full_sTDA_mv, sTDA_mv, AV, residual,
                                V_basis, W_basis, sub_A):
    V = ab_initio_V #ab-initio eigenkets
    lambda_matrix = np.diag(ab_initio_E)

    if args.AV == True:
        ab_initio_AV = AV
    else:
        ab_initio_AV = ab_initio_E * V



    '''commutator norm
        |V.T[A, A^se]V|
       =[λ, V.T*A^se*V]
    '''
    VAseV = np.dot(V.T, full_sTDA_mv(V))
    commutator = mathlib.commutator(lambda_matrix, VAseV)
    commutator_norm = np.linalg.norm(commutator)

    '''condition number
        |V.T * A^{se,-1}A * V|
        AV = λ * V
        A^{se,-1} AV = X => A^se X = AV
    '''
    with HiddenPrints():
        '''A^se X=W
        '''
        X = stapol_sTDDFT_initprec(Pr=-ab_initio_AV,
                                  tol=1e-6,
                matrix_vector_product=sTDA_mv)
    sub_condition = np.dot(V.T, X)
    condition_number = mathlib.cond_number(sub_condition)

    '''commutator_sub_A norm
        |[V_b.T A V_b, V_b.T A^se V_b]|
       =[sub_A, V_b Ase Vb]
    '''
    AseVb = full_sTDA_mv(V_basis)
    VbAseVb = np.dot(V_basis.T, AseVb)
    commutator_sub_A = mathlib.commutator(sub_A, VbAseVb)
    commutator_sub_A_norm = np.linalg.norm(commutator_sub_A)

    '''commutator basis V norm
        here, V_b is the Kryove space basis, W = AV
        |V_b.T [A, A^se] V_b|
       = V_b.T (A A^se - A^se A) V_b
       = V_b.T A A^se V_b - V_b.T A^se A V_b
       = W_b.T W^se - W^se.T W_b
    '''
    WbAseVb = np.dot(W_basis.T, AseVb)
    commutator_basis_A_norm = np.linalg.norm(WbAseVb - WbAseVb.T)

    ''' quality norm
        |(I - A^{se,-1}A) V |
        = V - X
    '''
    norm_0 = np.linalg.norm(V - X)

    ''' V.T quality norm
        | V.T (I - A^{se,-1}A) V|
        = I - V.TX
        = I - sub_condition
    '''
    norm_1 = np.linalg.norm(np.eye(args.nstates) - np.dot(V.T, X))

    ''' conditioner number of
        |V^T (A^{se}-λ)^{-1}(A - λ) V|
        = V^T(A^{se}-λ)^{-1}R
        = V^T V^new
    '''
    with HiddenPrints():
        ''' sTDA preconditioning R
        '''
        V_new, NA = sTDA_preconditioner(residual=residual,
                                  sub_eigenvalue=ab_initio_E,
                                             tol=1e-6,
                           matrix_vector_product=sTDA_mv,
                                             max=40)
    VV_new = np.dot(V.T, V_new)
    condition_number_1 = mathlib.cond_number(VV_new)

    ''' conditioner number of
        | V^T (A^{se}-λ)^{-1}A V |
        = V^T(A^{se}-λ)^{-1}λV
        = V^T V^new1
    '''
    with HiddenPrints():
        ''' sTDA preconditioning λV
        '''
        V_new1, NA = sTDA_preconditioner(residual=ab_initio_AV,
                                   sub_eigenvalue=ab_initio_E,
                                              tol=1e-6,
                            matrix_vector_product=sTDA_mv,
                                              max=40)

    VV_new1 = np.dot(V.T, V_new1)
    condition_number_2 = mathlib.cond_number(VV_new1)

    metric_list = [
        beta,
        alpha,
        commutator_norm,
        condition_number,
        commutator_sub_A_norm,
        commutator_basis_A_norm,
        norm_0,
        norm_1,
        condition_number_1,
        condition_number_2]

    return metric_list

def gen_metrics_diag(ab_initio_V, ab_initio_E, residual, NORM):
    V = ab_initio_V #ab-initio eigenkets
    lambda_matrix = np.diag(ab_initio_E)
    lambdaV = ab_initio_E * V

    '''commutator diag
        ||V.T[A, D]V||
       =[λ, V.T*D*V]
    '''
    DV = hdiag.reshape(-1,1)*V
    VDV = np.dot(V.T, DV)
    commutator_diag = np.dot(lambda_matrix,VDV)-np.dot(VDV,lambda_matrix)
    commutator_norm = np.linalg.norm(commutator_diag)

    '''condition number diag
        ||V.T * D^{-1}A * V||
        = V.T * D^{-1}  λ * V
    '''
    with HiddenPrints():
        ''' D^{-1} λ * V = X
            DX = λ * V
        '''
        X_D = stapol_A_diag_initprec(-ab_initio_AV)
    sub_condition = np.dot(V.T, X_D)
    s,u = np.linalg.eig(sub_condition)
    s = abs(s)
    condition_number = max(s)/min(s)

    ''' quality norm diag
        ||(I - D^{-1}A) * V||
        = V - X_D
    '''
    if NORM == True:
        X_D = X_D/np.linalg.norm(X_D)
    norm_0 = np.linalg.norm(V - X_D)

    ''' quality norm
        ||V.T (I - D^{-1}A) * V||
        = I - V.T X_D
        = I - sub_condition
    '''

    norm_1 = np.linalg.norm(np.eye(args.nstates) - np.dot(V.T, X_D))

    ''' quality preconditioner
        ||[I - (D-λ)^{-1}(A - λ)] * V||
        = V - (D-λ)^{-1}R
        = V - V^new
        (V^new = (D-λ)^{-1}R)
    '''
    with HiddenPrints():
        ''' sTDA preconditioning R
        '''
        V_new = TDA_A_diag_preconditioner(residual=residual,
                                    sub_eigenvalue=ab_initio_E,
                                             hdiag=hdiag)
    if NORM == True:
        V_new = V_new/np.linalg.norm(V_new)
    # norm_2 = np.linalg.norm(V - V_new_normed)
    norm_2 = np.linalg.norm(V - V_new)


    ''' quality preconditioner
        ||V^T [I - (D-λ)^{-1}(A - λ)] * V||
        = I - V^T(D-λ)^{-1}R
        = I - V^T V^new
    '''

    # norm_3 = np.linalg.norm(np.eye(args.nstates)-np.dot(V.T,V_new_normed))
    norm_3 = np.linalg.norm(np.eye(args.nstates)-np.dot(V.T,V_new))

    metric_list = \
    [commutator_norm, condition_number, norm_0, norm_1, norm_2, norm_3]

    return metric_list

if __name__ == "__main__":
    calc = gen_calc()
    TDA_combo = [            # option
    ['sTDA','sTDA'],         # 0
    ['Adiag','Adiag'],       # 1
    ['Adiag','sTDA'],        # 2
    ['sTDA','Adiag'],        # 3
    ['sTDA','Jacobi'],       # 4
    ['Adiag','Jacobi'],      # 5
    ['Adiag','new_ES'],      # 6
    ['sTDA','new_ES']]       # 7
    TDDFT_combo = [          # option
    ['sTDDFT','sTDDFT'],     # 0
    ['Adiag','Adiag'],       # 1
    ['Adiag','sTDDFT'],      # 2
    ['sTDDFT','Adiag']]      # 3
    print('|-------- In-house Developed {0} Starts ---------|'.format(calc))
    print('Residual conv =', args.conv_tolerance)
    if args.TDA == True:
        for option in args.ip_options:
            init, prec = TDA_combo[option]
            print('\n','Number of excited states = ', args.nstates)
            (Excitation_energies, eigenkets, AV, residual, Davidson_dic,
                                V_basis, W_basis, sub_A) = Davidson(init,prec)
            print('Excited State energies (eV) = ','\n', Excitation_energies)
            dump_yaml(Davidson_dic, calc, init, prec)

    if args.traceAA == True:
        metric_name_list = [
        'beta',
        'alpha',
        '|V.T[A,A^se]V|',
        'k(V.T(A^se,-1 A)V)',
        '|[a, Vb.TA^seVb]|',
        '|Vb.T[A, A^se]Vb|',
        '|(I - A^se,-1 A)V|',
        '|V.T(I-A^se,-1 A)V|',
        'k(V.T(A^se-λ)^-1(A-λ)V)',
        'k(V.T(A^se-λ)^-1 AV)']
        metric_name_format = "{0[0]:<8s}{0[1]:<8s}{0[2]:<25s}{0[3]:<25s}{0[4]:<20s}{0[5]:<25s}{0[6]:<25s}{0[7]:<25s}{0[8]:<25s}{0[9]:<25s}"
        metric_value_format = "{0[0]:<8.2f}{0[1]:<8.2f}{0[2]:<25.8f}{0[3]:<25.8f}{0[4]:<25.8f}{0[5]:<25.8f}{0[6]:<25.8f}{0[7]:<25.8f}{0[8]:<25.8f}{0[9]:<25.8f}"

        with open("data.txt", "w") as data_file:
            print(metric_name_format.format(metric_name_list),file=data_file)
        print(metric_name_format.format(metric_name_list))

        smallest_commutator_norm = 10000
        opt_alpha, opt_beta = 0,0
        for beta in args.beta_list:
            for alpha in args.alpha_list:
                GammaJ, GammaK = gen_gammaJK(alpha=alpha, beta=beta)
                sTDA_mv, full_sTDA_mv, sTDDFT_mv, sTDDFT_stapol_mv = gen_mv_fly(
                                                                GammaJ=GammaJ,
                                                                GammaK=GammaK)

                metric_list = gen_metrics(beta=beta,
                                         alpha=alpha,
                                   ab_initio_V=eigenkets,
             ab_initio_E=Excitation_energies/parameterlib.Hartree_to_eV,
                                       V_basis=V_basis,
                                       W_basis=W_basis,
                                         sub_A=sub_A,
                                  full_sTDA_mv=full_sTDA_mv,
                                       sTDA_mv=sTDA_mv,
                                            AV=AV,
                                      residual=residual)

                with open("data.txt", "a") as data_file:
                    print(metric_value_format.format(metric_list), file=data_file)
                print(metric_value_format.format(metric_list))

    if args.TDDFT == True:
        for option in args.ip_options:
            init, prec = TDDFT_combo[option]
            print('\nNumber of excited states =', args.nstates)
            Excitation_energies,X,Y,Davidson_dic = TDDFT_eigen_solver(init,prec)
            print('Excited State energies (eV) =\n',Excitation_energies)
            dump_yaml(Davidson_dic, calc, init, prec)
    if args.dynpol == True:
        for option in args.ip_options:
            init,prec = TDDFT_combo[option]
            print('\nPerturbation wavelength omega (nm) =', args.dynpol_omega)
            alpha_omega, Davidson_dic = dynamic_polarizability(init,prec)
            print('Dynamic polarizability tensor alpha')
            dump_yaml(Davidson_dic, calc, init, prec)
            for i in range(len(args.dynpol_omega)):
                print(args.dynpol_omega[i],'nm')
                print(alpha_omega[i])
    if args.stapol == True:
        for option in args.ip_options:
            init,prec = TDDFT_combo[option]
            print('\n')
            tensor_alpha, Davidson_dic = static_polarizability(init,prec)
            print('Static polarizability tensor alpha')
            print(tensor_alpha)
            dump_yaml(Davidson_dic, calc, init, prec)
    if args.sTDA == True:
        X, energies = sTDA_eigen_solver(k=args.nstates, tol=args.conv_tolerance)
    if args.sTDDFT == True:
        energies,X,Y = sTDDFT_eigen_solver(k=args.nstates,tol=args.conv_tolerance)

    if args.pytd == True:
        TD.nstates = args.nstates
        TD.conv_tol = args.conv_tolerance
        TD.kernel()
        end = time.time()
    if args.verbose > 3:
        for key in vars(args):
            print(key,'=', vars(args)[key])
    print('|-------- In-house Developed {0} Ends ----------|'.format(calc))
