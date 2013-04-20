import trace
import copy
import random
import math
from collections import Counter


def distrib(computation, samplingFn, *samplerArgs):
	"""
	Compute the discrete distribution over the given computation
	Only appropriate for computations that return a discrete value
	"""
	hist = Counter()
	samps = samplingFn(computation, *samplerArgs)
	for s in samps:
		hist[s[0]] += 1
	flnumsamps = float(len(samps))
	for s in hist:
		hist[s] /= flnumsamps
	return hist


def expectation(computation, samplingFn, *samplerArgs):
	"""
	Compute the expected value of a computation.
	Only appropriate for computations whose return value overloads the += and / operators
	"""
	samps = samplingFn(computation, *samplerArgs)
	return mean(map(lambda s: s[0], samps))


def mean(values):
	"""
	Compute the mean of a set of values
	"""
	mean = values[0]
	for v in values[1:]:
		mean += v
	return mean / float(len(values))


def MAP(computation, samplingFn, *samplerArgs):
	"""
	Maximum a posteriori inference (returns the highest probability sample)
	"""
	samps = samplingFn(computation, *samplerArgs)
	maxelem = max(samps, key=lambda s: s[1])
	return maxelem[0]


def rejectionSample(computation):
	"""
	Rejection sample a result from computation that satsifies
	all conditioning expressions.
	"""
	tr = trace.newTrace(computation)
	return tr.returnValue


def _randomChoice(items):
	"""
	Like random.choice, but returns None if items is empty
	"""
	if len(items) == 0:
		return None
	else:
		return random.choice(items)


class RandomWalkKernel:
	"""
	MCMC transition kernel that takes random walks
	by tweaking a single variable at a time
	"""

	def __init__(self, structural=True, nonstructural=True):
		self.structural = structural
		self.nonstructural = nonstructural
		self.proposalsMade = 0
		self.proposalsAccepted = 0

	def next(self, currTrace):

		self.proposalsMade += 1
		name = _randomChoice(currTrace.freeVarNames(self.structural, self.nonstructural))

		# If we have no free random variables, then just run the computation
		# and generate another sample (this may not actually be deterministic,
		# in the case of nested query)
		if name == None:
			currTrace.traceUpdate()
			return currTrace
		# Otherwise, make a proposal for a randomly-chosen variable, probabilistically
		# accept it
		else:
			nextTrace, fwdPropLP, rvsPropLP = currTrace.proposeChange(name)
			fwdPropLP -= math.log(len(currTrace.freeVarNames(self.structural, self.nonstructural)))
			rvsPropLP -= math.log(len(nextTrace.freeVarNames(self.structural, self.nonstructural)))
			acceptThresh = nextTrace.logprob - currTrace.logprob + rvsPropLP - fwdPropLP
			if nextTrace.conditionsSatisfied and math.log(random.random()) < acceptThresh:
				self.proposalsAccepted += 1
				return nextTrace
			else:
				return currTrace

	def stats(self):
		print "Acceptance ratio: {0} ({1}/{2})".format(float(self.proposalsAccepted)/self.proposalsMade, \
													   self.proposalsAccepted, self.proposalsMade)


class LARJInterpolationTrace(object):
	"""
	Abstraction for the linear interpolation of two execution traces
	"""

	def __init__(self, trace1, trace2, alpha=0.0):
		self.trace1 = trace1
		self.trace2 = trace2
		self.alpha = alpha

	@property
	def logprob(self):
		return (1-self.alpha)*self.trace1.logprob + self.alpha*self.trace2.logprob

	@property
	def conditionsSatisfied(self):
		return self.trace1.conditionsSatisfied and self.trace2.conditionsSatisfied

	@property
	def returnValue(self):
		return trace2.returnValue

	def freeVarNames(self, structural=True, nonstructural=True):
		return list(set(self.trace1.freeVarNames(structural, nonstructural) + \
						self.trace2.freeVarNames(structural, nonstructural)))

	def proposeChange(self, varname):
		var1 = self.trace1.getRecord(varname)
		var2 = self.trace2.getRecord(varname)
		nextTrace = LARJInterpolationTrace(copy.deepcopy(self.trace1) if var1 else self.trace1, \
										   copy.deepcopy(self.trace2) if var2 else self.trace2, \
										   self.alpha)
		var1 = nextTrace.trace1.getRecord(varname)
		var2 = nextTrace.trace2.getRecord(varname)
		var = (var1 if var1 else var2)
		assert(not var.structural)		# We're only supposed to be making changes to non-structurals here
		propval = var.erp._proposal(var.val, var.params)
		fwdPropLP = var.erp._logProposalProb(var.val, propval, var.params)
		rvsPropLP = var.erp._logProposalProb(propval, var.val, var.params)
		if var1:
			var1.val = propval
			var1.logprob = var1.erp._logprob(var1.val, var1.params)
			nextTrace.trace1.traceUpdate()
		if var2:
			var2.val = propval
			var2.logprob = var2.erp._logprob(var2.val, var2.params)
			nextTrace.trace2.traceUpdate()
		return nextTrace, fwdPropLP, rvsPropLP


class LARJKernel:
	"""
	MCMC transition kernel that does reversible jumps
	using the LARJ algorithm.
	"""

	def __init__(self, diffusionKernel, annealSteps, jumpFreq=None):
		self.diffusionKernel = diffusionKernel
		self.annealSteps = annealSteps
		self.jumpFreq = jumpFreq
		self.jumpProposalsMade = 0
		self.jumpProposalsAccepted = 0
		self.diffusionProposalsMade = 0
		self.diffusionProposalsAccepted = 0
		self.annealingProposalsMade = 0
		self.annealingProposalsAccepted = 0

	def next(self, currTrace):

		numStruct = len(currTrace.freeVarNames(nonstructural=False))
		numNonStruct = len(currTrace.freeVarNames(structural=False))

		# If we have no free random variables, then just run the computation
		# and generate another sample (this may not actually be deterministic,
		# in the case of nested query)
		if numStruct + numNonStruct == 0:
			currTrace.traceUpdate()
			return currTrace
		# Decide whether to jump or diffuse
		structChoiceProb = (self.jumpFreq if self.jumpFreq else float(numStruct)/(numStruct + numNonStruct))
		if random.random() < structChoiceProb:
			# Make a structural proposal
			return self.jumpStep(currTrace)
		else:
			# Make a nonstructural proposal
			prevAccepted = self.diffusionKernel.proposalsAccepted
			nextTrace = self.diffusionKernel.next(currTrace)
			self.diffusionProposalsMade += 1
			self.diffusionProposalsAccepted += (self.diffusionKernel.proposalsAccepted - prevAccepted)
			return nextTrace

	def jumpStep(self, currTrace):
		
		self.jumpProposalsMade += 1
		oldStructTrace = copy.deepcopy(currTrace)
		newStructTrace = copy.deepcopy(currTrace)

		# Randomly choose a structural variable to change
		structVars = newStructTrace.freeVarNames(nonstructural=False)
		name = _randomChoice(structVars)
		var = newStructTrace.getRecord(name)
		origval = var.val
		propval = var.erp._proposal(var.val, var.params)
		fwdPropLP = var.erp._logProposalProb(var.val, propval, var.params)
		var.val = propval
		var.logprob = var.erp._logprob(var.val, var.params)
		newStructTrace.traceUpdate()
		oldNumVars = len(structVars)
		newNumVars = len(newStructTrace.freeVarNames(nonstructural=False))
		fwdPropLP += newStructTrace.newlogprob - math.log(oldNumVars)

		# We only actually do annealing if we have any non-structural variables and we're doing more than
		# zero annealing steps
		annealingLpRatio = 0.0
		if len(oldStructTrace.freeVarNames(structural=False)) + len(newStructTrace.freeVarNames(structural=False)) != 0  and \
		   self.annealSteps > 0:
		 	aStep = 0
		 	lerpTrace = LARJInterpolationTrace(oldStructTrace, newStructTrace)
		 	prevAccepted = self.diffusionKernel.proposalsAccepted
			while aStep < self.annealSteps:
				lerpTrace.alpha = float(aStep)/(self.annealSteps-1)
				annealingLpRatio += lerpTrace.logprob
				lerpTrace = self.diffusionKernel.next(lerpTrace)
				annealingLpRatio -= lerpTrace.logprob
				aStep += 1
			self.annealingProposalsMade += self.annealSteps
			self.annealingProposalsAccepted += (self.diffusionKernel.proposalsAccepted - prevAccepted)
			oldStructTrace = lerpTrace.trace1
			newStructTrace = lerpTrace.trace2

		# Finalize accept/reject decision
		var = newStructTrace.getRecord(name)
		rvsPropLP = var.erp._logProposalProb(propval, origval, var.params) + oldStructTrace.lpDiff(newStructTrace) - math.log(newNumVars)
		acceptanceProb = newStructTrace.logprob - currTrace.logprob + rvsPropLP - fwdPropLP + annealingLpRatio
		if newStructTrace.conditionsSatisfied and math.log(random.random()) < acceptanceProb:
			self.jumpProposalsAccepted += 1
			return newStructTrace
		else:
			return currTrace

	def stats(self):
		overallProposalsMade = self.jumpProposalsMade + self.diffusionProposalsMade
		overallProposalsAccepted = self.jumpProposalsAccepted + self.diffusionProposalsAccepted
		if self.diffusionProposalsMade > 0:
			print "Diffusion acceptance ratio: {0} ({1}/{2})".format(float(self.diffusionProposalsAccepted)/self.diffusionProposalsMade, \
																	 self.diffusionProposalsAccepted, self.diffusionProposalsMade)
		if self.jumpProposalsMade > 0:
			print "Jump acceptance ratio: {0} ({1}/{2})".format(float(self.jumpProposalsAccepted)/self.jumpProposalsMade, \
																self.jumpProposalsAccepted, self.jumpProposalsMade)
		if self.annealingProposalsMade > 0:
			print "Annealing acceptance ratio: {0} ({1}/{2})".format(float(self.annealingProposalsAccepted)/self.annealingProposalsMade, \
																	 self.annealingProposalsAccepted, self.annealingProposalsMade)
		print "Overall acceptance ratio: {0} ({1}/{2})".format(float(overallProposalsAccepted)/overallProposalsMade, \
													 		   overallProposalsAccepted, overallProposalsMade)


def mcmc(computation, kernel, numsamps, lag=1, verbose=False):
	"""
	Do MCMC for 'numsamps' iterations using a given transition kernel
	"""
	currentTrace = trace.newTrace(computation)
	samps = []
	i = 0
	iters = numsamps * lag
	while i < iters:
		currentTrace = kernel.next(currentTrace)
		if i % lag == 0:
			samps.append((currentTrace.returnValue, currentTrace.logprob))
		i += 1
	if verbose:
		kernel.stats()
	return samps


def traceMH(computation, numsamps, lag=1, verbose=False):
	"""
	Sample from a probabilistic computation for some
	number of iterations using single-variable-proposal
	Metropolis-Hastings
	"""
	return mcmc(computation, RandomWalkKernel(), numsamps, lag, verbose)


def LARJMH(computation, numsamps, annealSteps, jumpFreq=None, lag=1, verbose=False):
	"""
	Sample from a probabilistic computation using locally annealed
	reversible jump mcmc
	"""
	return mcmc(computation, \
				LARJKernel(RandomWalkKernel(structural=False), annealSteps, jumpFreq), \
				numsamps, lag, verbose)

