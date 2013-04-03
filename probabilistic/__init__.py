
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
Soft and hard constraints
"""
from trace import factor
from trace import condition


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