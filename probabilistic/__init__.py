import erp
import database

"""
Random variable generators
"""
flip = erp._FlipRandomPrimitive()


"""
Factor creation
"""
def factor(num):
	database._rvdb.addFactor(num)


"""
Inference procedures
"""
from inference import sample