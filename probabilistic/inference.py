from database import _rvdb as rvdb
import copy
import random
import math


def sample(computation, iters):
	"""
	Sample from a probabilistic computation for some
	number of iterations.

	This uses vanilla, single-variable trace MH
	TODO: Implement other inference backends.
	"""
	global rvdb

	# Run computation to populate the database
	# with an initial trace
	rvdb.traceUpdate(computation)

	samps = []

	# MH inference loop
	for i in range(iters):

		# Make proposal for a randomly-chosen variable
		name, var = rvdb.chooseVariableRandomly()
		propval = var.erp._proposal(var.val, var.params)
		forwardPropLogProb = var.erp._logProposalProb(var.val, propval, var.params)
		reversePropLogProb = var.erp._logProposalProb(propval, var.val, var.params)

		# Copy the database, make the proposed change, and update the trace
		propdb = copy.deepcopy(rvdb)
		vrec = propdb.getRecord(name)
		vrec.val = propval
		retval = propdb.traceUpdate(computation)

		# Accept or reject the proposal
		if math.log(random.random()) < propdb.logprob - rvdb.logprob + reversePropLogProb - forwardPropLogProb:
			rvdb = propdb
		samps.append(retval)

	return samps
