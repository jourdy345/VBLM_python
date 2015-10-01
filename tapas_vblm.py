from __future__ import division
import os
import csv
import pandas as pd
import numpy as np
from scipy.special import gamma, gammaln

def check_input(y, X, **prior):
  if not (hasattr(y, '__len__') and not isinstance(y, str)):
    raise TypeError("y must be a list")
  if len(y) != np.array(y).shape[0]:
    raise TypeError("y and X must have same number of rows")
  if len(np.matrix(X).shape) != 2:
    raise TypeError("X must be a vector or a matrix")
  if not np.isscalar(prior['a_0']):
    raise TypeError("a_0 must be a scalar")
  if not np.isscalar(prior['b_0']):
    raise TypeError("b_0 must be a scalar")
  if not np.isscalar(prior['c_0']):
    raise TypeError("c_0 must be a scalar")
  if not np.isscalar(prior['d_0']):
    raise TypeError("d_0 must be a scalar")
  if not prior['a_0'] > 0:
    raise ValueError("a_0 must be positive")
  if not prior['b_0'] > 0:
    raise ValueError("b_0 must be positive")
  if not prior['c_0'] > 0:
    raise ValueError("c_0 must be positive")
  if not prior['d_0'] > 0:
    raise ValueError("d_0 must be positive")
  return y


def digamma(*args):
  if len(args) == 1:
    h = 0.00001
    gamdash = (gamma(args[0]-2.*h)-8.*gamma(args[0]-h)+8.*gamma(args[0]+h)-gamma(args[0]+2.*h))/12./h
    return float(gamdash)/gamma(args[0])
  elif len(args) > 2 or len(args) < 1:
    raise ValueError("There should be two arguments.")
  else:
    for arg in args:
      if type(arg) != float:
        raise TypeError("Arguments should be floating points.")
    gamdash = (gamma(args[0]-2.*args[1])-8.*gamma(args[0]-args[1])+8.*gamma(args[0]+args[1])-gamma(args[0]+2.*args[1]))/12./args[1]
    return float(gamdash)/gamma(args[0])

def free_energy(q, y, X, **kwargs):
  n = np.matrix(X).shape[0]
  d = np.matrix(X).shape[1]
  prior = kwargs

  # Expected log joint <ln p(y, beta, alpha, lambda)>_q
  J = (n/2. * (digamma(*[q['c_n']]) - np.log(q['d_n'])) - n/2. * np.log(2.*np.pi) - 0.5*q['c_n']/q['d_n']*np.dot(y, y) + q['c_n']/q['d_n']*(np.mat(q['mu_n'], dtype = 'f8')*np.mat(X, dtype = 'f8').T*np.mat(y, dtype = 'f8').T) - 0.5*q['c_n']/q['d_n']*np.trace(np.mat(X, dtype = 'f8').T*np.mat(X, dtype = 'f8')*(np.mat(q['mu_n'], dtype = 'f8').T*np.mat(q['mu_n'], dtype = 'f8') + np.linalg.inv(np.mat(q['Lambda_n'], dtype = 'f8')))) - d/2.*np.log(2.*np.pi) + n/2.*(digamma(*[q['a_n']])-np.log(q['b_n'])) - 0.5*q['a_n']/q['b_n'] * (np.dot(np.array(q['mu_n'], dtype = 'f8'), np.array(q['mu_n'], dtype = 'f8')) + np.trace(np.linalg.inv(np.mat(q['Lambda_n'], dtype = 'f8')))) + prior['a_0']*np.log(prior['b_0']) - gammaln(prior['a_0']) + (prior['a_0']-1.) * (digamma(*[q['a_n']])- np.log(q['b_n'])) - prior['b_0']*q['a_n']/q['b_n'] + prior['c_0']*np.log(prior['d_0']) - gammaln(prior['c_0']) + (prior['c_0']-1.)*(digamma(*[q['c_n']])-np.log(q['d_n'])) - prior['d_0']*q['c_n']/q['d_n'])

  # Entropy H[q]
  H = d/2 * (1+np.log(2*np.pi)) + 1/2*np.log(np.linalg.det(np.linalg.inv(np.mat(q['Lambda_n'])))) + q['a_n'] - np.log(q['b_n']) + gammaln(q['a_n']) + (1-q['a_n'])*digamma(*[q['a_n']]) + q['c_n'] - np.log(q['d_n']) + gammaln(q['c_n']) + (1-q['c_n'])*digamma(*[q['c_n']])

  # Free energy
  F = J + H

  return F

def invert_model(y, X, **kwargs):
  n, d = np.mat(X).shape
  prior = kwargs
  q = {}
  q_trace = []
  q['mu_n'] = np.zeros(d)
  q['Lambda_n'] = np.eye(d)
  q['a_n'] = prior['a_0']
  q['b_n'] = prior['b_0']
  q['c_n'] = prior['c_0']
  q['d_n'] = prior['d_0']
  q['F'] = -np.inf
  q['prior'] = prior

  q_trace.append(q)
  q_trace[0]['F'] = free_energy(q, y, X, **prior)
  # q_trace[0]['q'] = q
  # q_trace[0]['q']['F'] = free_energy(q, y, X, prior)

  # Variational algorithm
  max_iter = 30
  for i in xrange(max_iter):

    # (1) Update q(beta)
    q['Lambda_n'] = q['a_n']/q['b_n'] + q['c_n']/q['d_n'] * (np.mat(X).T*np.mat(X)) # a_n/b_n + c_n/d_n * X'X
    q['mu_n'] = q['c_n']/q['d_n'] * (np.linalg.inv(np.mat(q['Lambda_n'])))*((np.mat(X).T*np.mat(y).T)) #c_n/d_n * (λ_n)^(-1)X'y

    # (2) Update q(alpha)
    q['a_n'] = prior['a_0'] + d/2
    q['b_n'] = prior['b_0'] + 1/2 * (np.mat(q['mu_n'])*np.mat(q['mu_n']).T + np.trace(np.linalg.inv(np.mat(q['Lambda_n']))))

    # (3) Update q(lambda)
    q['c_n'] = prior['c_0'] + n/2
    q['d_n'] = (prior['d_0'] + 1/2*np.mat(y)*np.mat(y).T - np.mat(q['mu_n'])*np.mat(X).T*np.mat(y).T
                + 1/2*np.mat(q['mu_n'])*np.mat(X).T*np.mat(X)*np.mat(q['mu_n']).T)

    # Compute free energy
    F_old = q['F']
    q['F'] = free_energy(q, y, X, prior)

    # Keep record of intermediate results
    # q_trace[i+1]['q'] = q
    q_trace.append(q)

    # Convergence?
    if q['F'] - F_old < 10e-4:
      break
    if i == max_iter:
      print('Reached maximum number of iterations!')

  return q, q_trace

def tapas_vblm(y, X, *args):
  a_0 = args[0]
  b_0 = args[1]
  c_0 = args[2]
  d_0 = args[3]

  prior = {}
  prior['a_0'] = a_0
  prior['b_0'] = b_0
  prior['c_0'] = c_0
  prior['d_0'] = d_0

  # check input
  y = check_input(y, X, **prior)

  # Invert full model
  q, q_trace = invert_model(y, X, **prior)

  return q, q_trace





os.chdir('/Users/daeyounglim/Desktop/Computer_related/Variational/linearmodel/VBLM_python/')
with open('iris.csv', 'r') as iris:
  iris_data_iter = csv.reader(iris, delimiter = ',', quotechar = '"')
  iris_data = [ data for data in iris_data_iter ]
iris_data = np.asarray(iris_data)
y = iris_data[1:, 1]
X = iris_data[1:, 2:5]
y = np.asarray(y, dtype = 'f8')
X = np.asarray(X, dtype = 'f8')

prior_list = np.array([1., 1., 1., 1.], dtype = 'f8')
q, q_trace = tapas_vblm(y, X, *prior_list)
print q, q_trace


'''
Matlab code

function dg=digamma(x,h)
if(nargin==1);h=0.00001;end
gamdash=(gamma(x-2*h)-8*gamma(x-h)+8*gamma(x+h)-gamma(x+2*h))/12/h;
dg=gamdash./gamma(x);

function [q, q_trace] = invert_model(y, X, prior)

  # Data shortcuts
  [n,d] = size(X); # observations x regressors
  
  # Initialize variational posterior
  q.mu_n     = zeros(d,1);
  q.Lambda_n = eye(d);
  q.a_n      = prior.a_0;
  q.b_n      = prior.b_0;
  q.c_n      = prior.c_0;
  q.d_n      = prior.d_0;
  q.F        = -inf;
  q.prior    = prior;
  
  # Initialize trace of intermediate results?
  if nargout >= 2
      q_trace(1).q = q;
      q_trace(1).q.F = free_energy(q,y,X,prior);
  end
  
  # Variational algorithm
  nMaxIter = 30;
  for i = 1:nMaxIter
      
      # (1) Update q(beta)
      q.Lambda_n = q.a_n/q.b_n + q.c_n/q.d_n * ((X')*X);
      q.mu_n = q.c_n/q.d_n * (q.Lambda_n \ ((X')*y));
      
      # (2) Update q(alpha)
      q.a_n = prior.a_0 + d/2;
      q.b_n = prior.b_0 + 1/2 * (q.mu_n'*q.mu_n + trace(inv(q.Lambda_n)));
      
      # (3) Update q(lambda)
      q.c_n = prior.c_0 + n/2;
      q.d_n = prior.d_0 + 1/2*(y')*y -q.mu_n'*X'*y +1/2*q.mu_n'*(X')*X*q.mu_n;
      
      # Compute free energy
      F_old = q.F;
      q.F = free_energy(q,y,X,prior);
      
      # Append to trace of intermediate results?
      if nargout >= 2, q_trace(i+1).q = q; end
      
      # Convergence?
      if (q.F - F_old < 10e-4), break; end
      if (i == nMaxIter), warning('tapas_vblm: reached "%"d iterations',nMaxIter); end   # Quotation mark around percent sign added to prevent it from running
  end
end


## Computes the free energy of the model given the data.
function F = free_energy(q,y,X,prior)
    # Data shortcuts
    n = size(X, 1);
    d = size(X, 2);
    
    # Expected log joint <ln p(y,beta,alpha,lambda)>_q
    J = n/2*(digamma(q.c_n)-log(q.d_n)) - n/2*log(2*pi) ...
        - 0.5*q.c_n/q.d_n*((y')*y) + q.c_n/q.d_n*((q.mu_n')*(X')*y) ...
        - 0.5*q.c_n/q.d_n*trace((X')*X * (q.mu_n*q.mu_n' + inv(q.Lambda_n))) ...
      - d/2*log(2*pi) + n/2*(digamma(q.a_n)-log(q.b_n)) ...
        - 0.5*q.a_n/q.b_n * (q.mu_n'*q.mu_n + trace(inv(q.Lambda_n))) ...
      + prior.a_0*log(prior.b_0) - gammaln(prior.a_0) ...
        + (prior.a_0-1)*(digamma(q.a_n)-log(q.b_n)) - prior.b_0*q.a_n/q.b_n ...
      + prior.c_0*log(prior.d_0) - gammaln(prior.c_0) ...
        + (prior.c_0-1)*(digamma(q.c_n)-log(q.d_n)) - prior.d_0*q.c_n/q.d_n;
    
    # Entropy H[q]
    H = d/2*(1+log(2*pi)) + 1/2*log(det(inv(q.Lambda_n))) ...
      + q.a_n - log(q.b_n) + gammaln(q.a_n) + (1-q.a_n)*digamma(q.a_n) ...
      + q.c_n - log(q.d_n) + gammaln(q.c_n) + (1-q.c_n)*digamma(q.c_n);

    # Free energy
    F = J + H;
end

# ------------------------------------------------------------------------------
# Checks input validity.
function y = check_input(y,X,prior)
    assert(isvector(y), 'y must be a vector');
    y = y(:);
    assert(length(y)==size(X, 1), 'y and X must have same number of rows');
    assert(ndims(X) == 2, 'X must be a vector or a matrix');
    assert(isscalar(prior.a_0), 'a_0 must be a scalar');
    assert(isscalar(prior.b_0), 'b_0 must be a scalar');
    assert(isscalar(prior.c_0), 'c_0 must be a scalar');
    assert(isscalar(prior.d_0), 'd_0 must be a scalar');
    assert(prior.a_0 > 0, 'a_0 must be positive');
    assert(prior.b_0 > 0, 'b_0 must be positive');
    assert(prior.c_0 > 0, 'c_0 must be positive');
    assert(prior.d_0 > 0, 'd_0 must be positive');
end
'''