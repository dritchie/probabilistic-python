
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
Factor creation
"""
from database import factor


"""
Inference procedures
"""
from inference import sample


"""
Control structures
"""
from control import prfor, prwhile, prmap