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
	tr = trace.newTrace()
	return tr.rejectionInitialize(computation)


def _randomChoice(items):
	"""
	Like random.choice, but returns None if items is empty
	"""
	if len(items) == 0:
		return None
	else:
		return random.choice(items)


def _mhstep(computation, currSamp, currTrace, structural=True, nonstructural=True):
	"""
	One step of single-variable proposal Metropolis-Hastings
	Returns the next sample, the next trace, and a
	boolean indicating whether the proposal was accepted
	"""

	# Return values
	nextTrace = copy.deepcopy(currTrace)
	samp = currSamp
	accepted = False

	name = _randomChoice(currTrace.freeVarNames(structural, nonstructural))

	# If we have no free random variables, then just run the computation
	# and generate another sample (this may not actually be deterministic,
	# in the case of nested query)
	if name == None:
		samp = nextTrace.traceUpdate(computation)
		accepted = True
	# Otherwise, make a proposal for a randomly-chosen variable
	else:
		var = nextTrace.getRecord(name)
		propval = var.erp._proposal(var.val, var.params)
		fwdPropLP = var.erp._logProposalProb(var.val, propval, var.params)
		rvsPropLP = var.erp._logProposalProb(propval, var.val, var.params)

		# Make the proposed change, and update the trace
		var.val = propval
		var.logprob = var.erp._logprob(var.val, var.params)
		retval = nextTrace.traceUpdate(computation)

		# Accept or reject the proposal
		fwdPropLP += nextTrace.newlogprob - math.log(len(currTrace.freeVarNames(structural, nonstructural)))
		rvsPropLP += nextTrace.oldlogprob - math.log(len(nextTrace.freeVarNames(structural, nonstructural)))
		acceptThresh = nextTrace.logprob - currTrace.logprob + rvsPropLP - fwdPropLP
		if nextTrace.conditionsSatisfied and math.log(random.random()) < acceptThresh:
			accepted = True
			samp = retval
		else:
			nextTrace = currTrace
	return samp, nextTrace, accepted


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
	tr = trace.newTrace()
	currsamp = tr.rejectionInitialize(computation)

	# MH inference loop
	samps = [(currsamp, tr.logprob)]
	i = 0
	iters = numsamps * lag
	while i < iters:

		i += 1
		proposalsMade += 1
		currsamp, tr, accepted = _mhstep(computation, currsamp, tr)
		if accepted:
			proposalsAccepted += 1

		# Record the most recent sample
		if i % lag == 0:
			samps.append((currsamp, tr.logprob))

	if verbose:
		print "Acceptance ratio: {0}".format(float(proposalsAccepted)/proposalsMade)

	return samps


# def LARJMCMC(computation, numsamps, annealSteps=20, lag=1, verbose=False):
# 	"""
# 	Sample from a probabilistic computation using locally annealed
# 	reversible jump mcmc
# 	"""
	
# 	# Analytics
# 	jumpProposalsMade = 0
# 	jumpProposalsAccepted = 0
# 	annealingProposalsMade = 0
# 	annealingProposalsAccepted = 0
# 	diffusionProposalsMade = 0
# 	diffusionProposalsAccepted = 0

# 	# Run computation to get an initial trace
# 	tr = trace.newTrace()
# 	currsamp = tr.rejectionInitialize(computation)

# 	# MH inference loop
# 	samps = [(currsamp, tr.logprob)]
# 	i = 0
# 	iters = numsamps * lag
# 	while i < iters:

# 		i += 1

# 		randVarRecord = tr.randomFreeVar()

# 		# If we have no free random variables, then just run the computation
# 		# and generate another sample (this may not actually be deterministic,
# 		# in the case of nested query)
# 		if randVarRecord == None:
# 			currsamp = tr.traceUpdate(computation)
# 		# Otherwise, make a proposal for a randomly-chosen variable
# 		else:
# 			name, var = randVarRecord
# 			# If this is a non-structural variable, do normal diffusion
# 			if not var.structural:
# 				pass
# 			# Otherwise, we need to execute a jump move
# 			else:
# 				pass

