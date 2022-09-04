#!/usr/bin/python

import numpy as np
import scipy
import time


def commutator(A,B):
    commu = np.dot(A,B) - np.dot(B,A)
    return commu

def cond_number(A):
    s,u = np.linalg.eig(A)
    s = abs(s)
    cond = max(s)/min(s)
    return cond

def matrix_power(S,a):
    '''X == S^a'''
    s,ket = np.linalg.eigh(S)
    s = s**a
    X = np.dot(ket*s,ket.T)
    return X

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

def TDDFT_subspace_liear_solver(a, b, sigma, pi, p, q, omega):
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
