import torch
import math

#==================================================================
# Elementary functions
#==================================================================

# tensor addition
def add(X, Y):
    return (X[0] + Y[0], X[1] + Y[1])

# tensor subtraction (X - Y)
def sub(X, Y):
    return (X[0] - Y[0], X[1] - Y[1])

# Hermitian transpose
def ht(X):
    return (X[0].t(), -X[1].t())

# tensor multiplication
def mm(X, Y):
    Z_re = torch.mm(X[0], Y[0]) - torch.mm(X[1], Y[1])
    Z_im = torch.mm(X[0], Y[1]) + torch.mm(X[1], Y[0])
    return (Z_re, Z_im) 

# scalar multiplication
def scalar_mul(a, X):
    return (a * X[0], a * X[1])

# matrix inverse
def inverse(X):
    X_re = X[0]
    X_im = X[1]
    X_re_inv = torch.inverse(X_re)
    tmp = torch.mm(X_im, X_re_inv)
    tmp = torch.mm(tmp, X_im)
    Z_re = torch.inverse(X_re + tmp)
    tmp = - torch.mm(X_re_inv, X_im)
    Z_im = torch.mm(tmp, Z_re)
    return (Z_re, Z_im)

# random matrix with normal distribution
def normal(m,n,stdv):
    Z_re = torch.normal(torch.zeros(m,n), std = stdv/math.sqrt(2.0))
    Z_im = torch.normal(torch.zeros(m,n), std = stdv/math.sqrt(2.0))
    return (Z_re, Z_im)

# pseudo inverse (Moore-Penrose)
def pseudo_inverse(X):
    m = X[0].size()[0]
    n = X[0].size()[1]
    X_ht = c_ht(X)
    tmp = c_inverse(c_mm(X_ht, X))
    tmp2 = c_inverse(c_mm(X, X_ht))
    if n < m:
        return c_mm(tmp, X_ht)
    else:
        return c_mm(X_ht, tmp2)
    
# Squared error 
def squared_error(X, Y):
    Z_re = X[0] - Y[0]
    Z_im = X[1] - Y[1]
    return ((Z_re**2).sum() + (Z_im**2).sum()).item()

# normalize the norm
def normalize(X):
    return (X[0]/torch.sqrt(X[0]**2 + X[1]**2), X[1]/torch.sqrt(X[0]**2 + X[1]**2))

# zero tensor
def zeros(m,n):
    return (torch.zeros(m,n), torch.zeros(m,n))

# L2-norm
def norm(X):
    return torch.sqrt(X[0]**2 + X[1]**2)

# hadamard product
def hadamard_prod(X, Y):
    return (X[0]*Y[0]-X[1]*Y[1],X[1]*Y[0]+X[0]*Y[1])

# trace
def trace(X):
    return (torch.trace(X[0]),torch.trace(X[1]))

# transpose
def t(X):
    return (X[0].t(),X[1].t())

# conjugate
def conj(X):
    return (X[0],-X[1])

#==================================================================
# for DFT matrices
#==================================================================

# DFT matrix
def dft(n):
    B_re = torch.zeros(n, n)
    B_im = torch.zeros(n, n)
    for i in range(n):
        for j in range(n):
            B_re[i][j] = math.cos(-2.0*math.pi*i*j/n)/math.sqrt(n)
            B_im[i][j] = math.sin(-2.0*math.pi*i*j/n)/math.sqrt(n)
    return (B_re, B_im)

# IDFT matrix
def idft(n):
    B_re = torch.zeros(n, n)
    B_im = torch.zeros(n, n)
    for i in range(n):
        for j in range(n):
            B_re[i][j] = math.cos(2.0*math.pi*i*j/n)/math.sqrt(n)
            B_im[i][j] = math.sin(2.0*math.pi*i*j/n)/math.sqrt(n)
    return (B_re, B_im)

#==================================================================
# for FFT functions 
#==================================================================

# FFT
def fft(x): # input is assumed to be a tensor of size mbs x n
    a = x[0]
    b = x[1]
    bs = a.size()[0]
    nu = a.size()[1]
    a2 = a.view(bs, 1, nu)
    a3 = torch.transpose(a2, 1, 2)
    b2 = b.view(bs, 1, nu)
    b3 = torch.transpose(b2, 1, 2).view(bs, nu, 1)
    x_in = torch.cat([a3,b3], dim=2)
    p = torch.fft(x_in, 1, normalized=True)
    out_re = p[:,:,0].view(bs, nu)
    out_im = p[:,:,1].view(bs, nu)
    return (out_re, out_im)

# Inverse FFT
def ifft(x): # input is assumed to be a tensor of size mbs x n
    a = x[0]
    b = x[1]
    bs = a.size()[0]
    nu = a.size()[1]
    a2 = a.view(bs, 1, nu)
    a3 = torch.transpose(a2, 1, 2)
    b2 = b.view(bs, 1, nu)
    b3 = torch.transpose(b2, 1, 2).view(bs, nu, 1)
    x_in = torch.cat([a3,b3], dim=2)
    p = torch.ifft(x_in, 1, normalized=True)
    out_re = p[:,:,0].view(bs, nu)
    out_im = p[:,:,1].view(bs, nu)
    return (out_re, out_im)

#==================================================================
# for complex sparse matrices
#==================================================================

def to_dense(X): # X should be a sparse matrix
    return (X[0].to_dense(), X[1].to_dense())

def to_sparse(X): # X should be a dense matrix
    return (X[0].to_sparse(), X[1].to_sparse())

# scalar multiplication
def sp_scalar_mul(a, X): # a should be a scalar, X is a sparse matrix 
    return (a * X[0], a * X[1])

# Hermitian transpose
def sp_ht(X):
    return (X[0].t(), -1.0*X[1].t())


def to_device(X, device):
    return (X[0].to(device), X[1].to(device))