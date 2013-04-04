import trace
import copy
import random
import math
from collections import Counter


def distrib(computation, sampler, *samplerArgs):
	"""
	Compute the discrete distribution over the given computation
	Only appropriate for computations that return a discrete value
	"""
	hist = Counter()
	samps = sampler(computation, *samplerArgs)
	for s in samps:
		hist[s[0]] += 1
	flnumsamps = float(len(samps))
	for s in hist:
		hist[s] /= flnumsamps
	return hist

def expectation(computation, sampler, *samplerArgs):
	"""
	Compute the expected value of a computation.
	Only appropriate for computations whose return value overloads the += and / operators
	"""
	samps = sampler(computation, *samplerArgs)
	return mean(map(lambda s: s[0], samps))

def mean(values):
	"""
	Compute the mean of a set of values
	"""
	mean = values[0]
	for v in values[1:]:
		mean += v
	return mean / float(len(values))

def MAP(computation, sampler, *samplerArgs):
	"""
	Maximum a posteriori inference (returns the highest probability sample)
	"""
	samps = sampler(computation, *samplerArgs)
	maxelem = max(samps, key=lambda s: s[1])
	return maxelem[0]


def traceMH(computation, numsamps, lag=1, verbose=False):
	"""
	Sample from a probabilistic computation for some
	number of iterations using single-variable-proposal
	Metropolis-Hastings

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
	samps = [(currsamp, trace.getCurrentTrace().logprob)]
	i = 0
	iters = numsamps * lag
	while i < iters:

		i += 1

		tr = trace.getCurrentTrace()

		# Make proposal for a randomly-chosen variable
		name, var = tr.randomFreeVar()
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

		if i % lag == 0:
			samps.append((currsamp, trace.getCurrentTrace().logprob))

	trace.setCurrentTrace(None)

	if verbose:
		print "Acceptance ratio: {0}".format(float(proposalsAccepted)/proposalsMade)

	return samps
