from __future__ import division
from scipy.special import gamma, digamma
import theano
import theano.tensor as T
import numpy as np
import math
import csv
import os

os.chdir('/Users/daeyounglim/Desktop/Computer_related/Variational/linearmodel/VBLM_python/')
with open('iris.csv', 'r') as iris:
  iris_data_iter = csv.reader(iris, delimiter = ',', quotechar = '"')
  iris_data = [ data for data in iris_data_iter ]
iris_data = np.asarray(iris_data)
y = iris_data[1:, 1]
X = iris_data[1:, 2:5]
y = np.asarray(y, dtype = 'f8')
X = np.asarray(X, dtype = 'f8')
y_squared = np.asarray([ i**2 for i in y ])


u, v, t, s, f, g, h, j = T.dscalars('u', 'v', 't', 's', 'f', 'g', 'h', 'j')
f_b  = u + 1/2*((v+t)*(1/s + f**2) - 2*(v*g + h)*f + j + v * (g**2)) # b_N function that will be used in the update procedure
b_func = theano.function([u, v, t, s, f, g, h, j], f_b)
f_l = (u + v)*t/s # lambda_N function that will be used in the update procedure
lambda_func = theano.function([u, v, t, s], f_l)

a1, a2, a3, a4, a5, a6, a7 = T.dscalars('a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7')
# f_J = 1/2*(np.log(a3) + np.log(a2) + digamma(a1) - np.log(2*np.pi)) + a1*np.log(a2) - np.log(gamma(a1)) - (a3/2)*(a1/a2)*(a2/(a3*(a1-1))) - a1 - (a1 + (a4-1)/2)*(np.log(a2) + digamma(a1)) - 1/2*np.log(a3) -a1*np.log(a2) + np.log(gamma(a1)) + a1/(2*a2)*(a6 - 2*a5*a7 + a4*(a2/(a3*(a1-1)) + a5**2) + a3*(a2/(a3*(a1-1))) + 2*a2)
# objective_func = theano.function([a1, a2, a3, a4, a5, a6, a7], f_J)
# f_low = 1/2 * math.log(1/a3) + math.log(gamma(a1)) - a1*math.log(a2)
# lower_bound = theano.function([a1, a2, a3], f_low)

def lower_bound(a1, a2, a3):
  return 1/2 * math.log(1/a3) + math.log(gamma(a1)) - a1*math.log(a2)

sigma_yn = y.sum()
sigma_yn_squared = y_squared.sum()

N = len(y)
mu0 = y.sum() / N
lambda0 = 100
a0 = 5
b0 = 4


mu_N = (lambda0*mu0 + N*(y.sum()/N)) / (lambda0 + N)
x_bar = y.sum() / N
a_N = a0 + (N+1)/2

lambda_N = 1000
b_list = []
lambda_list = []
b_N_old = b_func(b0, lambda0, N, lambda_N, mu_N, mu0, sigma_yn, sigma_yn_squared)
steps = 0
# objective_old = objective_func(a0, b0, lambda0, N, mu0, sigma_yn_squared, sigma_yn)
objective_old = lower_bound(2, 2, 2)
while 1:
  lambda_N_old = lambda_func(lambda0, N, a_N, b_N_old)
  b_N_new = b_func(b0, lambda0, N, lambda_N, mu_N, mu0, sigma_yn, sigma_yn_squared)
  lambda_N_new = lambda_func(lambda0, N, a_N, b_N_new)
  b_list.append(b_N_new)
  lambda_list.append(lambda_N_new)
  # objective_new = objective_func(a0, b_N_new, lambda_N_new, N, mu0, sigma_yn_squared, sigma_yn)
  objective_new = lower_bound(a_N, b_N_new, lambda_N_new)
  steps += 1
  if abs(objective_new - objective_old) < 0.0001:
  # if steps == 20:
    break
  else:
    lambda_N_old = lambda_N_new
    b_N_old = b_N_new
    objective_old = objective_new



true_lambda_N = lambda0 + N
true_mu_N = (lambda0*mu0 + sigma_yn)/(lambda0 + N)
true_b_N = b0 + 1/2*((y - (y.sum()/N))**2).sum() + (lambda0*N*(((y.sum() / N) - mu0)**2))/(2*(lambda0 + N))
true_a_N = a0 + N/2
print('lambda_N ', lambda_N_new)
print('b_N ', b_N_new)
print('true b_N ', true_b_N)
print('true lambda_N ', true_lambda_N)
print('lambdas ', lambda_list)
print('bs', b_list)
print('How many steps? ', steps)

