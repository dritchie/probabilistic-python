import random
from database import _rvdb as rvdb

class _RandomPrimitive:
	"""
	Abstract base class for all ERPs
	"""

	def _sample_impl(self, params):
		pass

	def _sample(self, params):
		# Assumes _sample is called from __call__ in
		# conrete subclasses
		name = rvdb.currentName(numFrameSkip=2)
		val = rvdb.lookup(name, self, params)
		if val != None:
			return val
		else:
			val = self._sample_impl(params)
			rvdb.insert(name, self, params, val)
			return val

	def _logprob(self, val, params):
		pass

class _FlipRandomPrimitive(_RandomPrimitive):
	"""
	ERP with Bernoulli distribution
	"""

	def __init__(self):
		pass

	def __call__(self, p):
		return self._sample([p])

	def _sample_impl(self, params):
		p = params[0]
		randval = random.random()
		return (1 if randval < p else 0)

	def eval(self, binaryval, p):
		return (p if binaryval == 1 else 1.0-p)

	def _logprob(self, val, params):
		return self.eval(val, params[0])

# TODO: Implement more ERP types!