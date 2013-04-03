import trace
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

	# No nested query support
	assert(trace.getCurrentTrace() == None)

	# Analytics
	proposalsMade = 0
	proposalsAccepted = 0

	# Run computation to populate the database
	# with an initial trace
	currsamp = None
	conditionsSatisfied = False
	while not conditionsSatisfied:
		trace.newTrace()
		currsamp = trace.getCurrentTrace().traceUpdate(computation)
		conditionsSatisfied = trace.getCurrentTrace().conditionsSatisfied

	# Bail early if the computation is deterministic
	if trace.getCurrentTrace().numVars() == 0:
		return [currsamp for i in range(iters)]

	# MH inference loop
	samps = [currsamp]
	i = 0
	while i < iters:

		i += 1

		tr = trace.getCurrentTrace()

		# Make proposal for a randomly-chosen variable
		name, var = tr.chooseVariableRandomly()
		propval = var.erp._proposal(var.val, var.params)
		fwdPropLP = var.erp._logProposalProb(var.val, propval, var.params)
		rvsPropLP = var.erp._logProposalProb(propval, var.val, var.params)

		# Copy the database, make the proposed change, and update the trace
		currtrace = tr
		proptrace = copy.deepcopy(tr)
		trace.setCurrentTrace(proptrace)
		vrec = proptrace.getRecord(name)
		vrec.val = propval
		vrec.logprob = vrec.erp._logprob(vrec.val, vrec.params)
		retval = proptrace.traceUpdate(computation)

		# Accept or reject the proposal
		fwdPropLP += proptrace.newlogprob - math.log(currtrace.numVars())
		rvsPropLP += proptrace.oldlogprob - math.log(proptrace.numVars())
		acceptThresh = proptrace.logprob - currtrace.logprob + rvsPropLP - fwdPropLP
		if proptrace.conditionsSatisfied and math.log(random.random()) < acceptThresh:
			proposalsAccepted += 1
			currsamp = retval
		else:
			trace.setCurrentTrace(currtrace)
		proposalsMade += 1
		samps.append(currsamp)

	trace.setCurrentTrace(None)

	print "Acceptance ratio: {0}".format(float(proposalsAccepted)/proposalsMade)
	return samps
