import time
import numpy as np
import scipy
from opt_einsum import contract as einsum
import pyscf
from pyscf import gto, scf, dft, tddft, data, lib
import argparse
import os
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

def gen_args():
    parser = argparse.ArgumentParser(description='Davidson')
    parser.add_argument('-x', '--xyzfile',          type=str,   default='NA',  help='xyz filename (molecule.xyz)')
    parser.add_argument('-chk', '--checkfile',      type=bool,  default=False, help='checkpoint filename (.chk)')
    parser.add_argument('-m', '--method',           type=str,   default='RKS', help='RHF RKS UHF UKS')
    parser.add_argument('-f', '--functional',       type=str,   default='NA',  help='xc functional')
    parser.add_argument('-b', '--basis_set',        type=str,   default='NA',  help='basis set')
    parser.add_argument('-df', '--density_fit',     type=bool,  default=True,  help='density fitting turn on')
    parser.add_argument('-g', '--grid_level',       type=int,   default='3',   help='0-9, 9 is best')

    parser.add_argument('-n','--nstates',           type=int,   default = 4,      help='number of excited states')
    parser.add_argument('-pytd','--pytd',           type=bool,  default = False , help='whether to compare with PySCF TDDFT')

    parser.add_argument('-TDA','--TDA',             type=bool,  default = False, help='perform TDA')
    parser.add_argument('-TDDFT','--TDDFT',         type=bool,  default = False, help='perform TDDFT')
    parser.add_argument('-dynpol','--dynpol',       type=bool,  default = False, help='perform dynamic polarizability')
    parser.add_argument('-omega','--dynpol_omega',  type=float, default = [], nargs='+', help='dynamic polarizability with perurbation omega, a list')
    parser.add_argument('-stapol','--stapol',       type=bool,  default = False, help='perform static polarizability')
    parser.add_argument('-sTDA','--sTDA',           type=bool,  default = False, help='perform sTDA calculation')
    parser.add_argument('-sTDDFT','--sTDDFT',       type=bool,  default = False, help='perform sTDDFT calculation')
    parser.add_argument('-TT','--Truncate_test',    type=bool,  default = False, help='test the wall time for different virtual truncation')

    parser.add_argument('-TV','--truncate_virtual', type=float, default = 40,    help='the threshold to truncate virtual orbitals, in eV')

    parser.add_argument('-o','--ip_options',        type=int,   default = [0], nargs='+', help='0-7')
    parser.add_argument('-t','--conv_tolerance',    type=float, default= 1e-5, help='residual norm Convergence threhsold')

    parser.add_argument('-it','--initial_TOL',      type=float, default= 1e-3, help='conv for the inital guess')
    parser.add_argument('-pt','--precond_TOL',      type=float, default= 1e-2, help='conv for TDA preconditioner')

    parser.add_argument('-ei','--extrainitial',     type=int,   default= 8,    help='number of extral TDA initial guess vectors, 0-8')
    parser.add_argument('-max','--max',             type=int,   default= 30,   help='max iterations')

    parser.add_argument('-et','--eigensolver_tol',  type=float, default= 1e-5, help='conv for new guess generator in new_ES')
    parser.add_argument('-M','--memory',            type=int,   default= 4000, help='max_memory')
    parser.add_argument('-v','--verbose',           type=int,   default= 5,    help='mol.verbose = 3,4,5')

    parser.add_argument('-be','--beta',             type=float, default= [],    nargs='+', help='beta = 0.83')
    parser.add_argument('-al','--alpha',            type=float, default= [],    nargs='+', help='alpha = 0.83')

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

'''read xyz file and delete its first two lines'''
basename = args.xyzfile.split('.',1)[0]

def SCF_kernel():
    kernel_0 = time.time()
    f = open(args.xyzfile)
    atom_coordinates = f.readlines()
    del atom_coordinates[:2]
    '''build geometry in PySCF'''
    mol = gto.Mole()
    mol.atom = atom_coordinates
    mol.basis = args.basis_set
    mol.verbose = args.verbose
    mol.max_memory = args.memory
    print('mol.max_memory', mol.max_memory)
    mol.build(parse_arg = False)
    '''DFT or HF'''
    if args.method == 'RKS':
        mf = dft.RKS(mol)
    elif args.method == 'UKS':
        mf = dft.UKS(mol)
    elif args.method == 'RHF':
        mf = scf.RHF(mol)
    elif args.method == 'UHF':
        mf = scf.UHF(mol)
    if 'KS' in args.method:
        print('RKS')
        mf.xc = args.functional
        mf.grids.level = args.grid_level
    else:
        print('HF')
    if args.density_fit:
        mf = mf.density_fit()
        print('Density fitting turned on')
    if args.checkfile == True:
        '''use the *.chk file as scf input'''
        mf.chkfile = basename + '_' + args.functional + '.chk'
        mf.init_guess = 'chkfile'
    mf.conv_tol = 1e-10
    print ('Molecule built')
    print ('Calculating SCF Energy...')
    mf.kernel()

    kernel_1 = time.time()
    kernel_t = kernel_1 - kernel_0

    return mol, mf, kernel_t

mol, mf, kernel_t = SCF_kernel()

print ('SCF Done after %.2f'%kernel_t, 'seconds')

show_memory_info('after SCF')

'''Collect everything needed from PySCF'''

Hartree_to_eV = 27.211386245988

def gen_global_var():
    '''TDA_vind & TDDFT_vind are ab-initio matrix vector multiplication function
    '''
    td = tddft.TDA(mf)
    TD = tddft.TDDFT(mf)
    TDA_vind, hdiag = td.gen_vind(mf)
    TDDFT_vind, Hdiag = TD.gen_vind(mf)

    Natm = mol.natm
    '''mf.mo_occ is an array of occupance [2,2,2,2,2,0,0,0,0.....]
       N_bf is the total amount of MOs
       coefficient_matrix_C is the unorthonormalized coefficient matrix
       if no truncation, then max_vir = n_vir and n_occ + max_vir = N_bf
    '''
    mo_occ = mf.mo_occ
    coefficient_matrix_C = mf.mo_coeff
    N_bf = len(mo_occ)
    n_occ = len(np.where(mo_occ > 0)[0])
    n_vir = len(np.where(mo_occ == 0)[0])
    delta_hdiag = hdiag.reshape(n_occ, n_vir)
    A_size = n_occ * n_vir

    tol_eV = args.truncate_virtual/Hartree_to_eV
    homo_vir = delta_hdiag[-1,:]
    max_vir = len(np.where(homo_vir <= tol_eV)[0])

    max_vir_hdiag = delta_hdiag[:,:max_vir]

    A_reduced_size = n_occ * max_vir

    '''R_array is inter-particle distance array
       unit == ’Bohr’, 5.29177210903(80)×10^(−11) m
    '''
    R_array = pyscf.gto.mole.inter_distance(mol, coords=None)

    print('hdiag shape', hdiag.shape)
    print('n_occ = ', n_occ)
    print('n_vir = ', n_vir)
    print('max_vir = ', max_vir)
    print('A_size = ', A_size)
    print('A_reduced_size =', A_reduced_size)

    return TDA_vind, TDDFT_vind, hdiag, max_vir_hdiag, delta_hdiag,\
                    max_vir_hdiag, Natm, coefficient_matrix_C, N_bf, n_occ,\
                        n_vir, max_vir, A_size, A_reduced_size, R_array

TDA_vind, TDDFT_vind, hdiag, max_vir_hdiag, delta_hdiag,\
        max_vir_hdiag, Natm, coefficient_matrix_C, N_bf, n_occ,\
             n_vir, max_vir, A_size, A_reduced_size, R_array = gen_global_var()

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

def gen_HARDNESS():
    '''a dictionary of chemical hardness, by mappig two lists:
       list of elements 1-94
       list of hardness for elements 1-94, floats,in Hartree
    '''
    elements = ['H' , 'He', 'Li', 'Be', 'B' , 'C' , 'N' , 'O' , 'F' , 'Ne', \
    'Na', 'Mg', 'Al', 'Si', 'P' , 'S' , 'Cl', 'Ar', 'K' , 'Ca','Sc', 'Ti', \
    'V' , 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn','Ga', 'Ge', 'As', 'Se', \
    'Br', 'Kr', 'Rb', 'Sr', 'Y' , 'Zr','Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', \
    'Ag', 'Cd', 'In', 'Sn','Sb', 'Te', 'I' , 'Xe', 'Cs', 'Ba', 'La', 'Ce', \
    'Pr', 'Nd','Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',\
    'Lu', 'Hf', 'Ta', 'W' , 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg','Tl', 'Pb', \
    'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U' , 'Np', 'Pu']
    hardness = [0.47259288,0.92203391,0.17452888,0.25700733,0.33949086,\
    0.42195412,0.50438193,0.58691863,0.66931351,0.75191607,0.17964105,\
    0.22157276,0.26348578,0.30539645,0.34734014,0.38924725,0.43115670,\
    0.47308269,0.17105469,0.20276244,0.21007322,0.21739647,0.22471039,\
    0.23201501,0.23933969,0.24665638,0.25398255,0.26128863,0.26859476,\
    0.27592565,0.30762999,0.33931580,0.37235985,0.40273549,0.43445776,\
    0.46611708,0.15585079,0.18649324,0.19356210,0.20063311,0.20770522,\
    0.21477254,0.22184614,0.22891872,0.23598621,0.24305612,0.25013018,\
    0.25719937,0.28784780,0.31848673,0.34912431,0.37976593,0.41040808,\
    0.44105777,0.05019332,0.06762570,0.08504445,0.10247736,0.11991105,\
    0.13732772,0.15476297,0.17218265,0.18961288,0.20704760,0.22446752,\
    0.24189645,0.25932503,0.27676094,0.29418231,0.31159587,0.32902274,\
    0.34592298,0.36388048,0.38130586,0.39877476,0.41614298,0.43364510,\
    0.45104014,0.46848986,0.48584550,0.12526730,0.14268677,0.16011615,\
    0.17755889,0.19497557,0.21240778,0.07263525,0.09422158,0.09920295,\
    0.10418621,0.14235633,0.16394294,0.18551941,0.22370139]
    HARDNESS = dict(zip(elements,hardness))
    return HARDNESS

def matrix_power(S,a):
    '''X == S^a'''
    s,ket = np.linalg.eigh(S)
    s = s**a
    X = np.dot(ket*s,ket.T)
    return X

def orthonormalize(C):
    ''' produce orthonormalized coefficient matrix C, N_bf * N_bf
        S = mf.get_ovlp()  is basis overlap matrix
        S = np.dot(np.linalg.inv(c.T), np.linalg.inv(c))
        np.dot(C.T,C) is a an identity matrix
    '''
    S = mf.get_ovlp()
    X = matrix_power(S, 0.5)
    C = np.dot(X,C)
    return C

def gen_alpha_beta_ax():
    RSH_F = [
    'lc-b3lyp',
    'wb97',
    'wb97x',
    'wb97x-d3',
    'cam-b3lyp']
    RSH_paramt = [
    [0.53, 8.00, 4.50],
    [0.61, 8.00, 4.41],
    [0.56, 8.00, 4.58],
    [0.51, 8.00, 4.51],
    [0.38, 1.86, 0.90]]
    RSH_F_paramt = dict(zip(RSH_F, RSH_paramt))

    '''NA is for Hartree-Fork'''
    hybride_F = ['b3lyp', 'tpssh', 'm05-2x', 'pbe0', 'm06', 'm06-2x', 'NA']
    hybride_paramt = [0.2, 0.1, 0.56, 0.25, 0.27, 0.54, 1]
    Func_ax = dict(zip(hybride_F, hybride_paramt))

    beta1 = 0.2
    beta2 = 1.83
    alpha1 = 1.42
    alpha2 = 0.48
    '''RSH functionals have specific a_x, beta, alpha values;
       hybride fucntionals have fixed alpha12 and beta12 values,
       with different a_x values, by which create beta, alpha
    '''
    if args.functional in RSH_F:
        a_x, beta, alpha = RSH_F_paramt[args.functional]
    elif args.functional in hybride_F:
        a_x = Func_ax[args.functional]
        beta = beta1 + beta2 * a_x
        alpha = alpha1 + alpha2 * a_x

    if args.beta != []:
        beta = args.beta[0]

    if args.alpha != []:
        alpha = args.alpha[0]

    print('a_x =', a_x)
    print('beta =', beta)
    print('alpha =', alpha)

    return a_x, beta, alpha

a_x, beta, alpha = gen_alpha_beta_ax()

def gen_gammaJK():
    '''creat GammaK and GammaK matrix
       mol.atom_pure_symbol(atom_id) returns the element symbol
    '''
    HARDNESS = gen_HARDNESS()
    a = [HARDNESS[mol.atom_pure_symbol(atom_id)] for atom_id in range(Natm)]
    a = np.asarray(a).reshape(1,-1)
    eta = (a+a.T)/2
    GammaJ = (R_array**beta + (a_x * eta)**(-beta))**(-1/beta)
    GammaK = (R_array**alpha + eta**(-alpha)) **(-1/alpha)
    return GammaJ, GammaK

def generateQ():
    '''build q_iajb tensor'''
    C = orthonormalize(coefficient_matrix_C)
    aoslice = mol.aoslice_by_atom()
    q = np.zeros([Natm, N_bf, N_bf])
    for atom_id in range(Natm):
        shst, shend, atstart, atend = aoslice[atom_id]
        q[atom_id,:, :] = np.dot(C[atstart:atend, :].T, C[atstart:atend, :])
    return q

def gen_QJK(max_vir=max_vir):

    '''pre-calculate and store the Q-Gamma rank 3 tensor
       qia * gamma * qjb -> qia GK_q_jb
    '''
    Qstart = time.time()
    q_tensors = generateQ()
    GammaJ, GammaK = gen_gammaJK()

    q_ij = np.zeros((Natm, n_occ, n_occ))
    q_ij[:,:,:] = q_tensors[:,:n_occ,:n_occ]

    q_ab = np.zeros((Natm, max_vir, max_vir))
    q_ab[:,:,:] = q_tensors[:,n_occ:n_occ+max_vir,n_occ:n_occ+max_vir]

    q_ia = np.zeros((Natm, n_occ, max_vir))
    q_ia[:,:,:] = q_tensors[:,:n_occ,n_occ:n_occ+max_vir]

    GK_q_jb = einsum("Bjb,AB->Ajb", q_ia, GammaK)
    GJ_q_ab = einsum("Bab,AB->Aab", q_ab, GammaJ)
    Qend = time.time()
    Q_time = Qend - Qstart
    print('Q-Gamma tensors building time = %.2f'%Q_time)
    return q_ij, q_ab, q_ia , GK_q_jb, GJ_q_ab

q_ij, q_ab, q_ia , GK_q_jb, GJ_q_ab = gen_QJK()

show_memory_info('after Q matrix')

def gen_iajb_ijab_ibja_delta_fly(max_vir = max_vir, \
                                    q_ij = q_ij, \
                                    q_ab = q_ab, \
                                    q_ia = q_ia , \
                                 GK_q_jb = GK_q_jb, \
                                 GJ_q_ab = GJ_q_ab):
    '''define sTDA on-the-fly two electron intergeral (pq|rs)
       A_iajb * v = delta_ia_ia*v + 2(ia|jb)*v - (ij|ab)*v
       iajb_v = einsum('Aia,Bjb,AB,jbm -> iam', q_ia, q_ia, GammaK, V)
       ijab_v = einsum('Aij,Bab,AB,jbm -> iam', q_ij, q_ab, GammaJ, V)
    '''
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

    return iajb_fly, ijab_fly, ibja_fly, delta_fly, delta_max_vir_fly

iajb_fly, ijab_fly, ibja_fly, delta_fly, delta_max_vir_fly = \
                                                gen_iajb_ijab_ibja_delta_fly()

def gen_sTDA_sTDDFT_stapol_fly(max_vir = max_vir, \
                              iajb_fly = iajb_fly, \
                              ijab_fly = ijab_fly, \
                              ibja_fly = ibja_fly, \
                     delta_max_vir_fly = delta_max_vir_fly):

    def sTDA_mv(V):
        '''return AX'''
        V = V.reshape(n_occ, max_vir, -1)
        '''MV =  delta_fly(V) + 2*iajb_fly(V) - ijab_fly(V)'''
        MV = delta_max_vir_fly(V) + 2*iajb_fly(V) - ijab_fly(V)
        MV = MV.reshape(n_occ*max_vir,-1)
        return MV

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

    return sTDA_mv, sTDDFT_mv, sTDDFT_stapol_mv

sTDA_mv, sTDDFT_mv, sTDDFT_stapol_mv = gen_sTDA_sTDDFT_stapol_fly()

def Gram_Schmidt_bvec(A, bvec):
    '''orthonormalize vector b against all vectors in A
       b = b - A*(A.T*b)
       suppose A is orthonormalized
    '''
    if A.shape[1] != 0:
        projections_coeff = np.dot(A.T, bvec)
        bvec -= np.dot(A, projections_coeff)
    return bvec

def VW_Gram_Schmidt(x, y, V, W):
    '''orthonormalize vector |x,y> against all vectors in |V,W>'''
    m = np.dot(V.T,x)
    m += np.dot(W.T,y)

    n = np.dot(W.T,x)
    n += np.dot(V.T,y)

    x -= np.dot(V,m)
    x -= np.dot(W,n)

    y -= np.dot(W,m)
    y -= np.dot(V,n)
    return x, y

def Gram_Schmidt_fill_holder(V, count, vecs):
    '''V is a vectors holder
       count is the amount of vectors that already sit in the holder
       nvec is amount of new vectors intended to fill in the V
       count will be final amount of vectors in V
    '''
    nvec = np.shape(vecs)[1]
    for j in range(nvec):
        vec = vecs[:, j].reshape(-1,1)
        vec = Gram_Schmidt_bvec(V[:, :count], vec)   #single orthonormalize
        vec = Gram_Schmidt_bvec(V[:, :count], vec)   #double orthonormalize
        norm = np.linalg.norm(vec)
        if  norm > 1e-14:
            vec = vec/norm
            V[:, count] = vec[:,0]
            count += 1
    new_count = count
    return V, new_count

def S_symmetry_orthogonal(x,y):
    '''symmetrically orthogonalize the vectors |x,y> and |y,x>
       as close to original vectors as possible
    '''
    x_p_y = x + y
    x_p_y_norm = np.linalg.norm(x_p_y)

    x_m_y = x - y
    x_m_y_norm = np.linalg.norm(x_m_y)

    a = x_p_y_norm/x_m_y_norm

    x_p_y /= 2
    x_m_y *= a/2

    new_x = x_p_y + x_m_y
    new_y = x_p_y - x_m_y

    return new_x, new_y

def symmetrize(A):
    A = (A + A.T)/2
    return A

def anti_symmetrize(A):
    A = (A - A.T)/2
    return A

def check_orthonormal(A):
    '''define the orthonormality of a matrix A as the norm of (A.T*A - I)'''
    n = np.shape(A)[1]
    B = np.dot(A.T, A)
    c = np.linalg.norm(B - np.eye(n))
    return c

def VW_Gram_Schmidt_fill_holder(V_holder, W_holder, m, X_new, Y_new):
    '''put X_new into V, and Y_new into W
       m: the amount of vectors that already on V or W
       nvec: amount of new vectors intended to put in the V and W
    '''
    VWGSstart = time.time()
    nvec = np.shape(X_new)[1]

    GScost = 0
    normcost = 0
    symmetrycost = 0
    for j in range(0, nvec):
        V = V_holder[:,:m]
        W = W_holder[:,:m]

        x_tmp = X_new[:,j].reshape(-1,1)
        y_tmp = Y_new[:,j].reshape(-1,1)

        GSstart = time.time()
        x_tmp,y_tmp = VW_Gram_Schmidt(x_tmp, y_tmp, V, W)
        x_tmp,y_tmp = VW_Gram_Schmidt(x_tmp, y_tmp, V, W)
        GSend = time.time()
        GScost += GSend - GSstart

        symmetrystart = time.time()
        x_tmp,y_tmp = S_symmetry_orthogonal(x_tmp,y_tmp)
        symmetryend = time.time()
        symmetrycost += symmetryend - symmetrystart

        normstart = time.time()
        xy_norm = (np.dot(x_tmp.T, x_tmp)+np.dot(y_tmp.T, y_tmp))**0.5

        if  xy_norm > 1e-14:
            x_tmp = x_tmp/xy_norm
            y_tmp = y_tmp/xy_norm

            V_holder[:,m] = x_tmp[:,0]
            W_holder[:,m] = y_tmp[:,0]
            m += 1
        else:
            print('vector kicked out during GS orthonormalization')
        normend = time.time()
        normcost += normend - normstart

    VWGSend = time.time()
    VWGScost = VWGSend - VWGSstart
    # print('GScost',round(GScost/VWGScost *100, 2),'%')
    # print('normcost',round(normcost/VWGScost *100, 2),'%')
    # print('symmetrycost', round(symmetrycost/VWGScost *100, 2),'%')
    # print('check VW orthonormalization')
    # VW = np.vstack((V_holder[:,:m], W_holder[:,:m]))
    # WV = np.vstack((W_holder[:,:m], V_holder[:,:m]))
    # VWWV = np.hstack((VW,WV))
    # print('check_orthonormal VWWV:',check_orthonormal(VWWV))
    return V_holder, W_holder, m

def solve_AX_Xla_B(A, omega, Q):
    '''AX - XΩ  = Q
       A, Ω, Q are known, solve X
    '''
    Qnorm = np.linalg.norm(Q, axis=0, keepdims = True)
    Q /= Qnorm
    N_vectors = len(omega)
    a, u = np.linalg.eigh(A)
    ub = np.dot(u.T, Q)
    ux = np.zeros_like(Q)
    for k in range(N_vectors):
        ux[:, k] = ub[:, k]/(a - omega[k])
    X = np.dot(u, ux)
    X *= Qnorm

    return X

def TDA_A_diag_initial_guess(m, hdiag = hdiag):
    '''m is the amount of initial guesses'''
    hdiag = hdiag.reshape(-1,)
    V_size = hdiag.shape[0]
    Dsort = hdiag.argsort()
    energies = hdiag[Dsort][:m]*Hartree_to_eV
    V = np.zeros((V_size, m))
    for j in range(m):
        V[Dsort[j], j] = 1.0
    return V, energies

def TDA_A_diag_preconditioner(residual, sub_eigenvalue, current_dic = None,\
                            hdiag = hdiag, tol = None, full_guess=None, \
                        return_index=None, W_H=None, V_H=None, sub_A_H=None):
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

def sTDA_eigen_solver(k, tol=args.initial_TOL):
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
    V[:, :new_m],initial_energies = TDA_A_diag_initial_guess(\
                                            new_m, hdiag = max_vir_hdiag)
    for i in range(max):
        '''create subspace'''
        W[:, m:new_m] = sTDA_mv(V[:, m:new_m])
        sub_A = np.dot(V[:,:new_m].T, W[:,:new_m])
        sub_A = symmetrize(sub_A)

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
        new_guess = TDA_A_diag_preconditioner(\
                        residual = residual[:,index],\
                  sub_eigenvalue = sub_eigenvalue[:k][index],\
                           hdiag = max_vir_hdiag)

        '''orthonormalize the new guess against basis and put into V holder'''
        m = new_m
        V, new_m = Gram_Schmidt_fill_holder(V, m, new_guess)

    sTDA_D_end = time.time()
    sTDA_D = sTDA_D_end - sTDA_D_start
    print('sTDA A diagonalized in', i, 'steps; ', '%.4f'%sTDA_D, 'seconds' )
    print('threshold =', tol)
    print('sTDA excitation energies:')
    print(sub_eigenvalue[:k]*Hartree_to_eV)

    U = np.zeros((n_occ,n_vir,k))
    U[:,:max_vir,:] = full_guess.reshape(n_occ,max_vir,k)
    U = U.reshape(A_size, k)
    omega = sub_eigenvalue[:k]*Hartree_to_eV
    return U, omega

def sTDA_preconditioner(residual, sub_eigenvalue, tol=args.precond_TOL,\
                        current_dic=None, full_guess=None, return_index=None,\
                        W_H=None, V_H=None, sub_A_H=None):
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
    max = 30   # Maximum number of iterations

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
    Dp = np.where(abs(Dp)<t, \
                                        np.sign(Dp)*t, Dp)
    Dp = Dp.reshape(n_occ, n_vir, -1)
    D = Dp[:,:max_vir,:].reshape(A_reduced_size,-1)
    inv_D = 1/D

    '''generate initial guess'''
    Xig = P*inv_D
    count = 0
    V, new_count = Gram_Schmidt_fill_holder(V, count, Xig)

    origin_dic = current_dic.copy()
    current_dic['preconditioning'] = []
    mvcost = 0
    GScost = 0
    subcost = 0
    subgencost = 0

    for i in range(max):

        '''project sTDA_A matrix and vector P into subspace'''
        mvstart = time.time()
        W[:, count:new_count] = sTDA_mv(V[:, count:new_count])
        mvend = time.time()
        mvcost += mvend - mvstart

        substart = time.time()
        sub_P= np.dot(V[:,:new_count].T, P)
        sub_A = np.dot(V[:,:new_count].T, W[:,:new_count])
        subend = time.time()
        subgencost += subend - substart

        sub_A = symmetrize(sub_A)
        m = np.shape(sub_A)[0]

        substart = time.time()
        sub_guess = solve_AX_Xla_B(sub_A, omega, sub_P)
        subend = time.time()
        subcost += subend - substart

        full_guess = np.dot(V[:,:new_count], sub_guess)
        residual = np.dot(W[:,:new_count], sub_guess) - full_guess*omega - P

        r_norms = np.linalg.norm(residual, axis=0).tolist()
        current_dic['preconditioning'].append(\
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
        V, new_count = Gram_Schmidt_fill_holder(V, count, new_guess)
        GSend = time.time()
        GScost += GSend - GSstart

    p_end = time.time()
    p_cost = p_end - p_start

    if i == (max -1):
        print('_____sTDA Preconditioner Failed Due to Iteration Limit _______')
        print('failed after ', i, 'steps,', '%.4f'%p_cost,'s')
        print('orthonormality of V', check_orthonormal(V[:,:count]))
        print('current residual norms', r_norms)
    else:
        print('sTDA precond Done after', i, 'steps;', '%.4f'%p_cost,'seconds')

    print('max_norm = ', '%.2e'%max_norm)
    for enrty in ['subgencost', 'mvcost', 'GScost', 'subcost']:
        cost = locals()[enrty]
        print("{:<10} {:<5.4f}s  {:<5.2%}".format(enrty, cost, cost/p_cost))
    full_guess *= pnorm

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
    if current_dic != None:
        return U, origin_dic
    else:
        return U

def Jacobi_preconditioner(residual, sub_eigenvalue, current_dic, full_guess,\
                return_index = None, W_H = None, V_H = None, sub_A_H = None):
    '''(1-uu*)(A-Ω*I)(1-uu*)t = -B
       B is residual, we want to solve t
       z approximates t
       z = (A-Ω*I)^(-1)*(-B) - α(A-Ω*I)^(-1)*u
            let K_inv_y = (A-Ω*I)^(-1)*(-B)
            and K_inv_u = (A-Ω*I)^(-1)*u
       z = K_inv_y - α*K_inv_u
       where α = [u*(A-Ω*I)^(-1)y]/[u*(A-Ω*I)^(-1)u]
       first, solve (A-Ω*I)^(-1)y and (A-Ω*I)^(-1)u
    '''
    B = residual
    omega = sub_eigenvalue
    u = current_guess

    K_inv_y = sTDA_preconditioner(-B, omega)
    K_inv_u = sTDA_preconditioner(u, omega)
    n = np.multiply(u, K_inv_y).sum(axis=0)
    d = np.multiply(u, K_inv_u).sum(axis=0)
    Alpha = n/d
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

def new_ES(full_guess, return_index, W_H, V_H, sub_A_H, \
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

        new_guess = TDA_A_diag_preconditioner(residual[:,index], \
                                                    sub_eigenvalue[:k][index])
        V, new_m = Gram_Schmidt_fill_holder(V, m, new_guess)
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

def fill_dictionary(dic,init,prec,k,icost,pcost,wall_time,N_itr,N_mv,\
            initial_energies=None,energies=None,difference=None,overlap=None,\
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
        print('\nIteration ', ii)
        istart = time.time()

        MV_start = time.time()
        W[:, m:new_m] = TDA_matrix_vector(V[:,m:new_m])
        MV_end = time.time()
        iMVcost = MV_end - MV_start
        MVcost += iMVcost
        sub_A = np.dot(V[:,:new_m].T, W[:,:new_m])
        sub_A = symmetrize(sub_A)
        print('subspace size: ', np.shape(sub_A)[0])

        sub_eigenvalue, sub_eigenket = np.linalg.eigh(sub_A)
        full_guess = np.dot(V[:,:new_m], sub_eigenket[:, :k])
        residual = np.dot(W[:,:new_m], sub_eigenket[:,:k])
        residual -= full_guess * sub_eigenvalue[:k]

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
        V, new_m = Gram_Schmidt_fill_holder(V, m, new_guess)
        print('new generated guesses:', new_m - m)

        iend = time.time()
        icost = iend - istart
        current_dic['iteration cost'] = icost
        current_dic['iteration MV cost'] = iMVcost
        iteration_list[ii] = current_dic
        print('iMVcost %.4f'%iMVcost)
        print('icost %.4f'%icost)

    energies = sub_eigenvalue[:k]*Hartree_to_eV

    D_end = time.time()
    Dcost = D_end - D_start

    Davidson_dic = fill_dictionary(Davidson_dic, init=init, prec=prec, k=k,\
                                icost=init_time, pcost=Pcost, wall_time=Dcost,\
            energies = energies.tolist(), N_itr=ii+1, N_mv=np.shape(sub_A)[0],\
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
    return energies, full_guess, Davidson_dic

def TDDFT_A_diag_initial_guess(V_holder, W_holder, new_m, hdiag=hdiag):
    hdiag = hdiag.reshape(-1,)
    Dsort = hdiag.argsort()
    V_holder[:,:new_m], energies = TDA_A_diag_initial_guess(new_m, hdiag=hdiag)
    return V_holder, W_holder, new_m, energies,\
                V_holder[:,:new_m], W_holder[:,:new_m]

def TDDFT_A_diag_preconditioner(R_x, R_y, omega, hdiag=hdiag, tol=None):
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

def TDDFT_subspace_eigen_solver(a, b, sigma, pi, k):
    ''' [ a b ] x - [ σ   π] x  Ω = 0 '''
    ''' [ b a ] y   [-π  -σ] y    = 0 '''

    d = abs(np.diag(sigma))
    d_mh = d**(-0.5)

    s_m_p = d_mh.reshape(-1,1) * (sigma - pi) * d_mh.reshape(1,-1)

    '''LU = d^−1/2 (σ − π) d^−1/2'''
    ''' A = PLU '''
    ''' if A is diagonally dominant, P is identity matrix (in fact not always) '''
    P_permutation, L, U = scipy.linalg.lu(s_m_p)

    L = np.dot(P_permutation, L)

    L_inv = np.linalg.inv(L)
    U_inv = np.linalg.inv(U)

    ''' a ̃−b ̃= U^-T d^−1/2 (a−b) d^-1/2 U^-1 = GG^T '''
    dambd =  d_mh.reshape(-1,1)*(a-b)*d_mh.reshape(1,-1)
    GGT = np.linalg.multi_dot([U_inv.T, dambd, U_inv])

    G = scipy.linalg.cholesky(GGT, lower=True)
    G_inv = np.linalg.inv(G)

    ''' M = G^T L^−1 d^−1/2 (a+b) d^−1/2 L^−T G '''
    dapbd = d_mh.reshape(-1,1)*(a+b)*d_mh.reshape(1,-1)
    M = np.linalg.multi_dot([G.T, L_inv, dapbd, L_inv.T, G])

    omega2, Z = np.linalg.eigh(M)
    omega = (omega2**0.5)[:k]
    Z = Z[:,:k]

    ''' It requires Z^T Z = 1/Ω '''
    ''' x+y = d^−1/2 L^−T GZ Ω^-0.5 '''
    ''' x−y = d^−1/2 U^−1 G^−T Z Ω^0.5 '''

    x_p_y = d_mh.reshape(-1,1)\
            *np.linalg.multi_dot([L_inv.T, G, Z])\
            *(np.array(omega)**-0.5).reshape(1,-1)

    x_m_y = d_mh.reshape(-1,1)\
            *np.linalg.multi_dot([U_inv, G_inv.T, Z])\
            *(np.array(omega)**0.5).reshape(1,-1)

    x = (x_p_y + x_m_y)/2
    y = x_p_y - x

    return omega, x, y

def sTDDFT_preconditioner_subspace_eigen_solver(a, b, sigma, pi, p, q, omega):
    '''[ a b ] x - [ σ   π] x  Ω = p
       [ b a ] y   [-π  -σ] y    = q
       normalize the right hand side first
    '''
    pq = np.vstack((p,q))
    pqnorm = np.linalg.norm(pq, axis=0, keepdims = True)

    p /= pqnorm
    q /= pqnorm

    d = abs(np.diag(sigma))
    d_mh = d**(-0.5)

    '''LU = d^−1/2 (σ − π) d^−1/2
       A = PLU
       P is identity matrix only when A is diagonally dominant
    '''
    s_m_p = d_mh.reshape(-1,1) * (sigma - pi) * d_mh.reshape(1,-1)
    P_permutation, L, U = scipy.linalg.lu(s_m_p)
    L = np.dot(P_permutation, L)

    L_inv = np.linalg.inv(L)
    U_inv = np.linalg.inv(U)

    p_p_q_tilde = np.dot(L_inv, d_mh.reshape(-1,1)*(p+q))
    p_m_q_tilde = np.dot(U_inv.T, d_mh.reshape(-1,1)*(p-q))

    ''' a ̃−b ̃= U^-T d^−1/2 (a−b) d^-1/2 U^-1 = GG^T'''
    dambd = d_mh.reshape(-1,1)*(a-b)*d_mh.reshape(1,-1)
    GGT = np.linalg.multi_dot([U_inv.T, dambd, U_inv])

    '''G is lower triangle matrix'''
    G = scipy.linalg.cholesky(GGT, lower=True)
    G_inv = np.linalg.inv(G)

    '''a ̃+ b ̃= L^−1 d^−1/2 (a+b) d^−1/2 L^−T
       M = G^T (a ̃+ b ̃) G
    '''
    dapba = d_mh.reshape(-1,1)*(a+b)*d_mh.reshape(1,-1)
    a_p_b_tilde = np.linalg.multi_dot([L_inv, dapba, L_inv.T])
    M = np.linalg.multi_dot([G.T, a_p_b_tilde, G])
    T = np.dot(G.T, p_p_q_tilde)
    T += np.dot(G_inv, p_m_q_tilde * omega.reshape(1,-1))

    Z = solve_AX_Xla_B(M, omega**2, T)

    '''(x ̃+ y ̃) = GZ
       x + y = d^-1/2 L^-T (x ̃+ y ̃)
       x - y = d^-1/2 U^-1 (x ̃- y ̃)
    '''
    x_p_y_tilde = np.dot(G,Z)
    x_p_y = d_mh.reshape(-1,1) * np.dot(L_inv.T, x_p_y_tilde)

    x_m_y_tilde = (np.dot(a_p_b_tilde, x_p_y_tilde) - p_p_q_tilde)/omega
    x_m_y = d_mh.reshape(-1,1) * np.dot(U_inv, x_m_y_tilde)

    x = (x_p_y + x_m_y)/2
    y = x_p_y - x
    x *= pqnorm
    y *= pqnorm
    return x, y

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
    V_holder, W_holder, new_m, energies, Xig, Yig = \
    TDDFT_A_diag_initial_guess(V_holder, W_holder, new_m, hdiag = max_vir_hdiag)

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
        U1_holder[:, m:new_m], U2_holder[:, m:new_m] = \
                                        sTDDFT_mv(V[:, m:new_m], W[:, m:new_m])
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


        a = symmetrize(a)
        b = symmetrize(b)
        sigma = symmetrize(sigma)
        pi = anti_symmetrize(pi)

        subgenend = time.time()
        subgencost += subgenend - subgenstart

        '''solve the eigenvalue omega in the subspace'''
        subcost_start = time.time()
        omega, x, y = TDDFT_subspace_eigen_solver(a, b, sigma, pi, k)
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
        X_new, Y_new = TDDFT_A_diag_preconditioner(\
                R_x[:,index], R_y[:,index], omega[index], hdiag = max_vir_hdiag)

        '''GS and symmetric orthonormalization'''
        m = new_m
        GScost_start = time.time()
        V_holder, W_holder, new_m = \
                VW_Gram_Schmidt_fill_holder(V_holder, W_holder, m, X_new, Y_new)
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

    energies = omega*Hartree_to_eV
    print('sTDDFT excitation energy:')
    print(energies)
    return energies, X, Y

def sTDDFT_initial_guess(V_holder, W_holder, new_m):
    energies, X_new_backup, Y_new_backup = sTDDFT_eigen_solver(new_m)
    V_holder, W_holder, new_m = VW_Gram_Schmidt_fill_holder\
                            (V_holder, W_holder, 0,  X_new_backup, Y_new_backup)
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

    X_new, Y_new  = TDDFT_A_diag_preconditioner(\
                        P, Q, omega, hdiag = max_vir_hdiag)
    V_holder, W_holder, new_m = VW_Gram_Schmidt_fill_holder(\
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
        U1_holder[:, m:new_m], U2_holder[:, m:new_m] = \
                                sTDDFT_mv(V[:, m:new_m], W[:, m:new_m])
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

        a = symmetrize(a)
        b = symmetrize(b)
        sigma = symmetrize(sigma)
        pi = anti_symmetrize(pi)

        '''solve the x & y in the subspace'''
        subcost_start = time.time()
        x, y = sTDDFT_preconditioner_subspace_eigen_solver(\
                                    a, b, sigma, pi, p, q, omega)
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
        X_new, Y_new = TDDFT_A_diag_preconditioner(R_x[:,index], R_y[:,index],\
                                            omega[index], hdiag = max_vir_hdiag)
        Pend = time.time()
        Pcost += Pend - Pstart

        '''GS and symmetric orthonormalization'''
        m = new_m
        GS_start = time.time()
        V_holder, W_holder, new_m = VW_Gram_Schmidt_fill_holder(\
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

    X_full *=  pqnorm
    Y_full *=  pqnorm

    X = np.zeros((n_occ,n_vir,k))
    Y = np.zeros((n_occ,n_vir,k))

    X[:,:max_vir,:] = X_full.reshape(n_occ,max_vir,k)
    Y[:,:max_vir,:] = Y_full.reshape(n_occ,max_vir,k)

    if max_vir < n_vir:
        P2 = Rx[:,max_vir:,:].reshape(n_occ*(n_vir-max_vir),-1)
        Q2 = Ry[:,max_vir:,:].reshape(n_occ*(n_vir-max_vir),-1)

        X2, Y2 = TDDFT_A_diag_preconditioner(\
                        P2, Q2, omega, hdiag = delta_hdiag[:,max_vir:])
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

        a = symmetrize(a)
        b = symmetrize(b)
        sigma = symmetrize(sigma)
        pi = anti_symmetrize(pi)

        print('subspace size: %s' %sigma.shape[0])

        omega, x, y = TDDFT_subspace_eigen_solver(a, b, sigma, pi, k)

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
        V_holder, W_holder, new_m = VW_Gram_Schmidt_fill_holder(\
                                        V_holder, W_holder, m, X_new, Y_new)
        print('m & new_m', m, new_m)
        if new_m == m:
            print('All new guesses kicked out during GS orthonormalization')
            break

    omega *= Hartree_to_eV

    difference = np.mean((np.array(initial_energies) - np.array(omega))**2)
    difference = float(difference)

    overlap = float(np.linalg.norm(np.dot(X_ig.T, X_full)) \
                    + np.linalg.norm(np.dot(Y_ig.T, Y_full)))

    TDDFT_end = time.time()
    TDDFT_cost = TDDFT_end - TDDFT_start

    Davidson_dic = fill_dictionary(Davidson_dic, init=init, prec=prec, k=k,\
            icost=init_time, pcost=Pcost, wall_time=TDDFT_cost, \
            energies=omega.tolist(), N_itr=ii+1, N_mv=np.shape(sigma)[0], \
            initial_energies=initial_energies, difference=difference,\
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
    P_origin[:,:] = P[:,:]

    pnorm = np.linalg.norm(P, axis=0, keepdims = True)
    pqnorm = pnorm * (2**0.5)
    print('pqnorm', pqnorm)
    P /= pqnorm

    P = np.tile(P,k)
    Q = P

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
    X_p_Y *= np.tile(pqnorm,k)
    for jj in range(k):
        '''*-1 from the definition of dipole moment. *2 for double occupancy'''
        X_p_Y_tmp = X_p_Y[:,3*jj:3*(jj+1)]
        alpha_omega_ig.append(np.dot(P_origin.T, X_p_Y_tmp)*-2)
    print('initial guess of tensor alpha')
    for i in range(k):
        print(args.dynpol_omega[i],'nm')
        print(alpha_omega_ig[i])

    V_holder, W_holder, new_m = VW_Gram_Schmidt_fill_holder(\
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
        x, y = sTDDFT_preconditioner_subspace_eigen_solver(\
                    a, b, sigma, pi, -p, -q, omega)
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
        X_new, Y_new = new_guess_generator(R_x[:,index], \
                            R_y[:,index], omega[index], tol=args.precond_TOL)
        Pend = time.time()
        Pcost += Pend - Pstart

        m = new_m
        GS_start = time.time()
        V_holder, W_holder, new_m = VW_Gram_Schmidt_fill_holder(\
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

    X_p_Y *= np.tile(pqnorm,k)

    for jj in range(k):
        X_p_Y_tmp = X_p_Y[:,3*jj:3*(jj+1)]
        alpha_omega.append(np.dot(P_origin.T, X_p_Y_tmp)*-2)

    difference = 0
    for i in range(k):
        difference += np.mean((alpha_omega_ig[i] - alpha_omega[i])**2)

    difference = float(difference)

    show_memory_info('Total Dynamic polarizability')
    Davidson_dic = fill_dictionary(Davidson_dic, init=init, prec=prec, k=3*k,\
            icost=initial_cost, pcost=Pcost, wall_time=dp_cost, \
            energies=omega.tolist(), N_itr=ii+1, N_mv=np.shape(sigma)[0], \
            difference=difference, overlap=overlap,\
            tensor_alpha=[i.tolist() for i in alpha_omega],\
            initial_tensor_alpha=[i.tolist() for i in alpha_omega_ig])
    return alpha_omega, Davidson_dic

def stapol_A_diag_initprec(P, hdiag=hdiag, tol=None):
    d = hdiag.reshape(-1,1)
    P = -P/d
    # P /= -d
    return P

def stapol_sTDDFT_initprec(Pr, tol=args.initial_TOL):
    '''(A* + B*)X = -P
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
    V_holder, new_m = Gram_Schmidt_fill_holder(V_holder, m, X_ig)
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
        U_holder[:, m:new_m] = sTDDFT_stapol_mv(V_holder[:,m:new_m])
        MV_end = time.time()
        MVcost += MV_end - MV_start

        V = V_holder[:,:new_m]
        U = U_holder[:,:new_m]

        subgenstart = time.time()
        p = np.dot(V.T, P)
        a_p_b = np.dot(V.T,U)
        a_p_b = symmetrize(a_p_b)

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
        V_holder, new_m = Gram_Schmidt_fill_holder(V_holder, m, X_new)
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

    X_full *= pnorm

    U = np.zeros((n_occ,n_vir,npvec))
    U[:,:max_vir,:] = X_full.reshape(n_occ,max_vir,-1)[:,:,:]

    if max_vir < n_vir:
        ''' DX2 = -P2'''
        P2 = Pr.reshape(n_occ,n_vir,-1)[:,max_vir:,:]
        P2 = P2.reshape(n_occ*(n_vir-max_vir),-1)
        D2 = hdiag.reshape(n_occ,n_vir,-1)[:,max_vir:,:]
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
       X_new = (residual - P)/D'''
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

    V_holder, new_m = Gram_Schmidt_fill_holder(V_holder, 0, X_ig)
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
        a_p_b = symmetrize(a_p_b)
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
            print('static polarizability precodure aborted\n')
            break

        '''preconditioning step'''
        Pstart = time.time()

        X_new = new_guess_generator(-residual[:,index], tol=args.precond_TOL)
        Pend = time.time()
        Pcost += Pend - Pstart

        '''GS and symmetric orthonormalization'''
        m = new_m
        GS_start = time.time()
        V_holder, new_m = Gram_Schmidt_fill_holder(V_holder, m, X_new)
        GS_end = time.time()
        GScost += GS_end - GS_start
        if new_m == m:
            print('All new guesses kicked out during GS orthonormalization')
            break

    X_full = np.dot(V,x)
    overlap = float(np.linalg.norm(np.dot(X_ig.T, X_full)))

    X_full *= pnorm

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
    Davidson_dic = fill_dictionary(Davidson_dic, init=init, prec=prec, k=3,\
            icost=initial_cost, pcost=Pcost, wall_time=sp_cost, \
            N_itr=ii+1, N_mv=np.shape(a_p_b)[0], difference=difference,\
            overlap=overlap, tensor_alpha=[i.tolist() for i in tensor_alpha],\
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
    for calc in ['TDA','TDDFT','dynpol','stapol',\
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
            print('\nNumber of excited states =', args.nstates)
            Excitation_energies, eigenkets, Davidson_dic = Davidson(init,prec)
            print('Excited State energies (eV) =\n',Excitation_energies)
            dump_yaml(Davidson_dic, calc, init, prec)
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
    if args.Truncate_test == True:
        n_states= args.nstates
        X = np.random.rand(A_size,n_state)
        Y = np.random.rand(A_size,n_state)
        print('n_vir = ', n_vir)
        print('A_size =', A_size)
        print('n_states =', n_states)
        print("{:<8} {:<8} {:<8} {:<8}".format(\
                'eV', 'max_vir', 'sTDA_t', 'sTDDFT_t'))
        for vir_trunc in [40, 50, 60, 70, 10000000]:
            del max_vir, sTDA_mv, sTDDFT_mv
            max_vir = gen_maxvir(tol_eV = vir_trunc)
            q_ij, q_ab, q_ia , GK_q_jb, GJ_q_ab = gen_QJK(max_vir=max_vir)
            # print('q_ab', q_ab.shape, 'GK_q_jb', GK_q_jb.shape)
            iajb_fly, ijab_fly, ibja_fly, delta_fly = gen_iajb_ijab_ibja_delta_fly(\
                                            max_vir=max_vir, \
                                            q_ij = q_ij, \
                                            q_ab = q_ab, \
                                            q_ia = q_ia , \
                                            GK_q_jb = GK_q_jb, \
                                            GJ_q_ab = GJ_q_ab)

            sTDA_mv, sTDDFT_mv, sTDDFT_stapol_mv = gen_sTDA_sTDDFT_stapol_fly(\
                                            max_vir=max_vir, \
                                            iajb_fly = iajb_fly, \
                                            ijab_fly = ijab_fly, \
                                            ibja_fly = ibja_fly, \
                                            delta_fly = delta_fly)

            sTDA_start = time.time()
            sTDA_X = sTDA_mv(X)
            sTDA_end = time.time()
            sTDA_mv_time = sTDA_end - sTDA_start

            sTDDFT_start = time.time()
            sTDDFT_X, sTDDFT_Y = sTDDFT_mv(X, Y)
            sTDDFT_end = time.time()
            sTDDFT_mv_time = sTDDFT_end - sTDDFT_start

            print("{:<8} {:<8} {:<8.4f} {:<8.4f}".format(\
                    vir_trunc, max_vir, sTDA_mv_time, sTDDFT_mv_time))
    if args.pytd == True:
        TD.nstates = args.nstates
        TD.conv_tol = args.conv_tolerance
        TD.kernel()
        end = time.time()
    if args.verbose > 3:
        for key in vars(args):
            print(key,'=', vars(args)[key])
    print('|-------- In-house Developed {0} Ends ----------|'.format(calc))
