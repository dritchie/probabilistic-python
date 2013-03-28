
"""
Random variable generators
"""
import erp
flip = erp._FlipRandomPrimitive()


"""
Factor creation
"""
import database
def factor(num):
	database.getCurrentDatabase().addFactor(num)


"""
Inference procedures
"""
from inference import sample