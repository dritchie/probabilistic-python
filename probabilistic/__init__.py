
"""
Random variable generators
"""
from erp import flip, gaussian, gamma, beta, binomial, poisson, dirichlet, multinomial, uniform, multinomialDraw, uniformDraw


"""
Hard and soft constraints
"""
from trace import condition, factor
def softEq(a, b, tolerance):
	return erp.gaussian_logprob(a-b, 0, tolerance)


"""
Inference procedures
"""
from inference import mean, distrib, expectation, MAP, rejectionSample, traceMH, LARJMH


"""
Control structures
"""
from control import ntimes, foreach, until, repeat


"""
Stochastic memoization
"""
from memoize import mem