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

	# Prepare return values
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
		print "Acceptance ratio: {0} ({1}/{2})".format(float(proposalsAccepted)/proposalsMade, proposalsAccepted, proposalsMade)

	return samps


def LARJMCMC(computation, numsamps, annealSteps=40, lag=1, verbose=False):
	"""
	Sample from a probabilistic computation using locally annealed
	reversible jump mcmc
	"""
	
	# Analytics
	jumpProposalsMade = 0
	jumpProposalsAccepted = 0
	annealingProposalsMade = [0]
	annealingProposalsAccepted = [0]
	diffusionProposalsMade = 0
	diffusionProposalsAccepted = 0

	# Run computation to get an initial trace
	tr = trace.newTrace()
	currsamp = tr.rejectionInitialize(computation)

	def jumpStep():

		# Prepare return values
		nextTrace = tr
		samp = currsamp
		accepted = False

		oldStructTrace = copy.deepcopy(tr)
		newStructTrace = copy.deepcopy(tr)

		# Randomly choose structural variable, make proposal, update trace
		structVars = newStructTrace.freeVarNames(nonstructural=False)
		name = _randomChoice(structVars)
		var = newStructTrace.getRecord(name)
		propval = var.erp._proposal(var.val, var.params)
		fwdPropLP = var.erp._logProposalProb(var.val, propval, var.params)
		rvsPropLP = var.erp._logProposalProb(propval, var.val, var.params)
		var.val = propval
		retval = newStructTrace.traceUpdate(computation)
		oldNumVars = len(structVars)
		newNumVars = len(newStructTrace.freeVarNames(nonstructural=False))
		fwdPropLP += newStructTrace.newlogprob - math.log(oldNumVars)

		# Prepare for annealing loop
		aStep = 0
		annealingLpRatio = 0.0
		lastAnnealingSamp = retval

		# We only actually do annealing if we have any non-structural variables
		if len(oldStructTrace.freeVarNames(structural=False)) + len(newStructTrace.freeVarNames(structural=False)) != 0:

			def annealingStep(alpha):

				# Prepare return values
				nextOldStructTrace = copy.deepcopy(oldStructTrace)
				nextNewStructTrace = copy.deepcopy(newStructTrace)
				samp = lastAnnealingSamp
				accepted = False

				# Pick random non-structural variable uniformly over all non-structurals in both
				# the old and the new structures
				nonStructName = _randomChoice(list(set(nextOldStructTrace.freeVarNames(structural=False) + \
													   nextNewStructTrace.freeVarNames(structural=False))))

				# Propose change to both the old and the new structures
				# (Or just one, if the chosen variable isn't shared)
				oldVar = nextOldStructTrace.getRecord(nonStructName)
				newVar = nextNewStructTrace.getRecord(nonStructName)
				nonNullVar = (oldVar if oldVar else newVar)
				propval = nonNullVar.erp._proposal(nonNullVar.val, nonNullVar.params)
				fwdPropLP = nonNullVar.erp._logProposalProb(nonNullVar.val, propval, nonNullVar.params)
				rvsPropLP = nonNullVar.erp._logProposalProb(propval, nonNullVar.val, nonNullVar.params)
				retval = lastAnnealingSamp
				if oldVar:
					oldVar.val = propval
					nextOldStructTrace.traceUpdate(computation)
				if newVar:
					newVar.val = propval
					retval = nextNewStructTrace.traceUpdate(computation)

				# Accept/reject this annealing move
				acceptThresh = ((1-alpha)*nextOldStructTrace.logprob + alpha*nextNewStructTrace.logprob + rvsPropLP) - \
							   ((1-alpha)*oldStructTrace.logprob + alpha*newStructTrace.logprob + fwdPropLP)
				if nextOldStructTrace.conditionsSatisfied and nextNewStructTrace.conditionsSatisfied and \
				   math.log(random.random()) < acceptThresh:
					samp = retval
					accepted = True
				else:
					nextOldStructTrace = oldStructTrace
					nextNewStructTrace = newStructTrace
				return samp, nextOldStructTrace, nextNewStructTrace, accepted

			while aStep < annealSteps:
				alpha = float(aStep)/(annealSteps-1)
				annealingLpRatio += (1-alpha)*oldStructTrace.logprob + alpha*newStructTrace.logprob
				lastAnnealingSamp, oldStructTrace, newStructTrace, accepted = annealingStep(alpha)
				annealingLpRatio -= (1-alpha)*oldStructTrace.logprob + alpha*newStructTrace.logprob
				annealingProposalsMade[0] += 1
				if accepted:
					annealingProposalsAccepted[0] += 1
				aStep += 1

		# Finalize the acceptance criterion and choose whether to accept
		rvsPropLP += oldStructTrace.lpDiff(newStructTrace) - math.log(newNumVars)
		acceptanceProb = newStructTrace.logprob - tr.logprob + rvsPropLP - fwdPropLP + annealingLpRatio
		if newStructTrace.conditionsSatisfied and math.log(random.random()) < acceptanceProb:
			nextTrace = newStructTrace
			accepted = True
			samp = lastAnnealingSamp
		return samp, nextTrace, accepted

	# Outer MH inference loop
	samps = [(currsamp, tr.logprob)]
	i = 0
	iters = numsamps * lag
	while i < iters:

		i += 1

		numStruct = len(tr.freeVarNames(nonstructural=False))
		numNonStruct = len(tr.freeVarNames(structural=False))

		# If we have no free random variables, then just run the computation
		# and generate another sample (this may not actually be deterministic,
		# in the case of nested query)
		if numStruct + numNonStruct == 0:
			currsamp = tr.traceUpdate(computation)
		# Otherwise, choose whether to modify a structural or nonstructural variable
		# (Mathematically equivalent to randomly sampling from the full set of variables)
		else:
			structChoiceProb = float(numStruct)/(numStruct + numNonStruct)
			if random.random() < structChoiceProb:
				# Make a structural proposal
				jumpProposalsMade += 1
				currsamp, tr, accepted = jumpStep()
				if accepted:
					jumpProposalsAccepted += 1
			else:
				# Make a nonstructural proposal
				diffusionProposalsMade += 1
				currsamp, tr, accepted = _mhstep(computation, currsamp, tr, structural=False)
				if accepted:
					diffusionProposalsAccepted += 1

		# Record the most recent sample
		if i % lag == 0:
			samps.append((currsamp, tr.logprob))

	overallProposalsMade = jumpProposalsMade + diffusionProposalsMade
	overallProposalsAccepted = jumpProposalsAccepted + diffusionProposalsAccepted
	if verbose:
		print "Diffusion acceptance ratio: {0} ({1}/{2})".format(float(diffusionProposalsAccepted)/diffusionProposalsMade, \
																 diffusionProposalsAccepted, diffusionProposalsMade)
		print "Jump acceptance ratio: {0} ({1}/{2})".format(float(jumpProposalsAccepted)/jumpProposalsMade, \
															jumpProposalsAccepted, jumpProposalsMade)
		print "Annealing acceptance ratio: {0} ({1}/{2})".format(float(annealingProposalsAccepted[0])/annealingProposalsMade[0], \
																 annealingProposalsAccepted[0], annealingProposalsMade[0])
		print "Overall acceptance ratio: {0} ({1}/{2})".format(float(overallProposalsAccepted)/overallProposalsMade, \
													 		   overallProposalsAccepted, overallProposalsMade)

	return samps

