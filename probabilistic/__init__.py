
"""
Random variable generators
"""
import erp
flip = erp.FlipRandomPrimitive()
gaussian = erp.GaussianRandomPrimitive()
gamma = erp.GammaRandomPrimitive()
beta = erp.BetaRandomPrimitive()
binomial = erp.BinomialRandomPrimitive()
poisson = erp.PoissonRandomPrimitive()
dirichlet = erp.DirichletRandomPrimitive()
multinomial = erp.MultinomialRandomPrimitive()
uniform = erp.UniformRandomPrimitive()


"""
Hard and soft constraints
"""
from trace import condition
from trace import factor
def softEq(a, b, tolerance):
	return erp.gaussian_logprob(a-b, 0, tolerance)


"""
Inference procedures
"""
from inference import distrib
from inference import expectation
from inference import MAP
from inference import traceMH


"""
Control structures
"""
from control import prfor, prwhile, prmap


"""
Stochastic memoization
"""
from memoize import mem