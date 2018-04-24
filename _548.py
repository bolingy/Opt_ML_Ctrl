
# packages
import numpy as np
import scipy as sp
from scipy import linalg as la
import control as ctrl
import matplotlib as mpl
import pylab as plt

from _547 import *

def PSD(n,sqrt=False):
  """
  compute random positive semidefinite matrix

  input:
    n - int - dimension of matrix
    (optional)
    sqrt - bool - whether to return S such that Q = np.dot( S.T, S)

  output:
    Q - n x n - Q = Q^T,  spec Q \subset R^+
  """
  H = np.random.randn(n,n)
  d,u = np.linalg.eig(H + H.T)
  S = np.dot( u, np.dot( np.diag( np.sqrt( d*np.sign(d) ) ), u.T ) )
  if sqrt:
    return np.dot(S.T, S), S
  else:
    return np.dot(S.T, S)

def steepest_descent(u,J=None,DJ=None,a=armijo):
  """
  execute one step of steepest descent:

    u' = u - a * DJ(u) 

  input:
    u - m array - base point
    J : R^m --> R - objective 
    (optional:)
    DJ : R^m --> R^m - gradient
    a  : (J,u,d) |--> scalar  - stepsize 

  output:
    u - a * DJ(u)
  """
  if DJ is None:
      assert J is not None, "J must be provided if DJ isn't"
      DJ = lambda u_ : D(J,u_)
  DJu = DJ(u)
  assert DJu.size == u.size, 'DJ(u) has size m = u.size'
  DJu.shape = (u.size,)
  return u - a(J,u,-DJu) * DJu # steepest descent step

def newton_raphson(u,J=None,DJ=None,D2J=None,asserts=True):
  """
  execute one step of Newton-Raphson iteration:

    u' = u - DJ(u) ( D^2 J(u) )^{-1}

  input:
    u - m array - base point
    (optional:)
    J : R^m --> R - objective - not needed if DJ provided
    DJ : R^m --> R^m - gradient
    D2J : R^m --> R^{m x m} - Hessian - must be symmetric
    (if DJ (and/or D2J) not provided:)
    DJ (and/or D2J) will be approximated with finite-central-differences

  output:
    u - np.dof(DJ(u), la.inv( D2J(u) ))
  """
  if DJ is None:
      assert J is not None, "J must be provided if DJ isn't"
      DJ = lambda u_ : D(J,u_)
  if D2J is None:
      D2J = lambda u_ : D(DJ,u_)
  DJu = DJ(u)
  if asserts:
    assert DJu.size == u.size, 'DJ(u) has size m = u.size'
    DJu.shape = (u.size,)
  D2Ju = D2J(u)
  if asserts:
    assert D2Ju.shape == (u.size,u.size), 'D2J(u) is m x m, m = u.size'
    assert np.allclose( D2Ju - D2Ju.T, 0. ), 'D2J(u) is symmetric'
    assert np.all( la.eigvals(D2Ju).real > 0 ), 'D2J(u) is positive-definite'
  return u - np.dot( DJu, la.inv(D2Ju) ) # Newton-Raphson step

