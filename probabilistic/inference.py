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


def rejectionSample(computation):
	"""
	Rejection sample a result from computation that satsifies
	all conditioning expressions.
	"""

	samp = None
	conditionsSatisfied = False
	while not conditionsSatisfied:
		tr = trace.newTrace()
		samp = tr.traceUpdate(computation)
		conditionsSatisfied = tr.conditionsSatisfied

	return samp


def traceMH(computation, numsamps, lag=1, verbose=False):
	"""
	Sample from a probabilistic computation for some
	number of iterations using single-variable-proposal
	Metropolis-Hastings
	"""

	# Analytics
	proposalsMade = 0
	proposalsAccepted = 0

	# Run computation to get an initial trace
	tr = None
	currsamp = None
	conditionsSatisfied = False
	while not conditionsSatisfied:
		tr = trace.newTrace()
		currsamp = tr.traceUpdate(computation)
		conditionsSatisfied = tr.conditionsSatisfied

	# MH inference loop
	samps = [(currsamp, tr.logprob)]
	i = 0
	iters = numsamps * lag
	while i < iters:

		i += 1

		randVarRecord = tr.randomFreeVar()

		# If we have no free random variables, then just run the computation
		# and generate another sample (this may not actually be deterministic,
		# in the case of nested query)
		if randVarRecord == None:
			currsamp = tr.traceUpdate(computation)
		# Otherwise, make a proposal for a randomly-chosen variable
		else:
			name, var = randVarRecord
			propval = var.erp._proposal(var.val, var.params)
			fwdPropLP = var.erp._logProposalProb(var.val, propval, var.params)
			rvsPropLP = var.erp._logProposalProb(propval, var.val, var.params)

			# Copy the database, make the proposed change, and update the trace
			currtrace = tr
			proptrace = copy.deepcopy(tr)
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
				tr = proptrace
			proposalsMade += 1

		# Record the most recent sample
		if i % lag == 0:
			samps.append((currsamp, tr.logprob))

	if verbose:
		print "Acceptance ratio: {0}".format(float(proposalsAccepted)/proposalsMade)

	return samps
