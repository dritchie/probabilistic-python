import random
import database
import math

class _RandomPrimitive:
	"""
	Abstract base class for all ERPs
	"""

	def _sample_impl(self, params):
		pass

	def _sample(self, params, conditionedValue=None):
		# Assumes _sample is called from __call__ in
		# conrete subclasses
		return database.lookupVariableValue(self, params, 2, conditionedValue)

	def _logprob(self, val, params):
		pass

	def _proposal(self, currval, params):
		"""
		Subclasses can override to do more efficient proposals
		"""
		return self._sample_impl(params)

	def _logProposalProb(self, currval, propval, params):
		"""
		Subclasses can override to do more efficient proposals
		"""
		return self._logprob(propval, params)

class _FlipRandomPrimitive(_RandomPrimitive):
	"""
	ERP with Bernoulli distribution
	"""

	def __init__(self):
		pass

	def __call__(self, p, conditionedValue=None):
		return self._sample([p], conditionedValue)

	def _sample_impl(self, params):
		p = params[0]
		randval = random.random()
		return (1 if randval < p else 0)

	def _logprob(self, val, params):
		p = params[0]
		prob = (p if val == 1 else 1.0-p)
		return math.log(prob)

	def _proposal(self, currval, params):
		return int(not(currval))

	def _logProposalProb(self, currval, propval, params):
		return 0.0 		# There's only one way to flip a binary variable


# TODO: Implement more ERP types!