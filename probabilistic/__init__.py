
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
from database import factor
from database import condition


"""
Inference procedures
"""
from inference import sample


"""
Control structures
"""
from control import prfor, prwhile, prmap