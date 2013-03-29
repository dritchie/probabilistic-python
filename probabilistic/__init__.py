
"""
Random variable generators
"""
import erp
flip = erp._FlipRandomPrimitive()


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