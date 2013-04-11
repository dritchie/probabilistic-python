from probabilistic import *
import math
from collections import Counter

###############################

def distribForward(computation, iters):
	hist = Counter()
	i = 0
	while i < iters:
		i += 1
		hist[computation()] += 1
	for x in hist:
		hist[x] /= float(iters)
	return hist

def compareForwardToMHDists(computation, iters):
	forwardhist = distribForward(computation, iters)
	print "Forward hist:"
	print forwardhist
	mhhist = distrib(computation, traceMH, iters)
	print "MH hist:"
	print mhhist

def meanForward(computation, iters):
	mean = computation()
	i = 0
	while i < iters-1:
		i += 1
		mean += computation()
	return mean / iters

def compareForwardToMHMeans(computation, iters):
	forwardmean = meanForward(computation, iters)
	print "Forward mean: {0}".format(forwardmean)
	mhmean = expectation(computation, traceMH, iters)
	print "MH mean: {0}".format(mhmean)

###############################

def ones():
	if flip(0.75):
		return [1] + ones()
	else:
		return []

def numOnes():
	return len(ones())

def constrainedOnes():
	seq = ones()
	factor(-math.pow(abs(len(seq) - 4), 10))
	return seq

def sumOfTen():
	num = 0
	num += flip(0.5, conditionedValue=True)
	num += flip(0.5, conditionedValue=True)
	# num += flip(0.5)
	# num += flip(0.5)
	num += flip(0.5)
	num += flip(0.5)
	num += flip(0.5)
	num += flip(0.5)
	num += flip(0.5)
	num += flip(0.5)
	num += flip(0.5)
	num += flip(0.5)
	return num

def constrainedSumOfTen():
	num = sumOfTen()
	condition(num >= 5)
	return num

def sumOfTenWhile():
	num = [0]
	i = [0]
	def block():
		i[0] += 1
		num[0] += flip(0.5)
	until(lambda: i[0] == 10, block)
	return num[0]

def sumOfTenFor():
	num = [0]
	def block(i):
		num[0] += flip(0.5)
	foreach(xrange(10), block)
	return num[0]

def sumOfTenMap():
	return sum(map(lambda x: flip(0.5), range(10)))

def oneGaussian():
	return gaussian(10, 0.5)

def oneGamma():
	return gamma(9, 0.5)

def oneBeta():
	return beta(2, 2)

def oneBinomial():
	return binomial(0.5, 40)

def onePoisson():
	return poisson(10)

def memTest():
	func = mem(lambda x: flip(x))
	result1 = func(0.5)
	result2 = func(0.5)
	return result1 == result2

def noisyOr(a, astrength, b, bstrength, baserate):
	return (flip(astrength) and a) or \
		   (flip(bstrength) and b) or \
		   flip(baserate)

def sprinklerTest():

	weights = {"rain-str": 0.9, \
			   "rain-prior": 0.3, \
			   "sprinkler-str": 0.9, \
			   "sprinkler-prior": 0.2, \
			   "grass-baserate": 0.1}

	@mem
	def rain(day):
		return flip(weights["rain-prior"])

	@mem
	def sprinkler(day):
		return flip(weights["sprinkler-prior"])

	@mem
	def grassIsWet(day):
		return noisyOr(rain(day), weights["rain-str"], \
					   sprinkler(day), weights["sprinkler-str"], \
					   weights["grass-baserate"])

	condition(grassIsWet("day2"))

	return rain("day2")

## ChurchServ version of the above test, for comparison:
# (define (noisy-or a astrength b bstrength baserate)
#   (or (and (flip astrength) a)
#       (and (flip bstrength) b)
#       (flip baserate)))
# (define sprinklerTest
#   (mh-query 100 100
#      (define weight (lambda (ofwhat)
#        (case ofwhat
#          (('rain-str) 0.9)
#          (('rain-prior) 0.3)
#          (('sprinkler-str) 0.9)
#          (('sprinkler-prior) 0.2)
#          (('grass-baserate) 0.1))))
#      (define grass-is-wet (mem (lambda (day)
#        (noisy-or
#         (rain day) (weight 'rain-str)
#         (sprinkler day) (weight 'sprinkler-str)
#         (weight 'grass-baserate)))))
#      (define rain (mem (lambda (day)
#        (flip (weight 'rain-prior)))))
#      (define sprinkler (mem (lambda (day)
#        (flip (weight 'sprinkler-prior)))))
     
#      (rain 'day2)
     
#      (grass-is-wet 'day2)
#   )
# )
# (hist sprinklerTest "Rained on Day2?")

# stringLengthProbs = [1.0/25, 2.0/25, 3.0/25, 4.0/25, 5.0/25, 4.0/25, 3.0/25, 2.0/25, 1.0/25]
# stringLengths = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
stringLengthProbs = [0.5, 0.5]
stringLengths = [2, 3]
penaltyMultiplier = 10

def stringsOfLength(length, numvals):
	def helper(n, seqSoFar):
		if n == 0:
			yield tuple(seqSoFar)
		else:
			for i in xrange(numvals):
				for tup in helper(n-1, seqSoFar + [i]):
					yield tup
	for tup in helper(length, []):
		yield tup

def constrainedStringA():
	numelems = multinomialDraw(stringLengths, stringLengthProbs, isStructural=True)
	seq = repeat(numelems, lambda: int(flip(0.5)))
	if numelems % 2 == 0:
		factor(-penaltyMultiplier * len(filter(lambda num: num == 1, seq)))
	else:
		factor(-penaltyMultiplier * len(filter(lambda num: num == 0, seq)))
	return tuple(seq)

def constrainedStringATrueDist():
	hist = {}
	for numelems in stringLengths:
		# Probability of choosing this many elements
		numlp = erp.multinomial_logprob(stringLengths.index(numelems), stringLengthProbs)
		for seq in stringsOfLength(numelems, 2):
			# Prior probability of each element value
			lp = -numelems*math.log(2)
			# Penalties
			if numelems % 2 == 0:
				lp -= penaltyMultiplier * len(filter(lambda num: num == 1, seq))
			else:
				lp -= penaltyMultiplier * len(filter(lambda num: num == 0, seq))
			hist[seq] = numlp + lp
	# Normalize by partition function
	logz = math.log(sum(map(lambda lp: math.exp(lp), hist.values())))
	for seq in hist:
		hist[seq] = math.exp(hist[seq] - logz)
	return hist


def constrainedStringB():
	onethird = 1.0/3
	numelems = multinomialDraw(stringLengths, stringLengthProbs, isStructural=True)
	seq = repeat(numelems, lambda: multinomial([onethird, onethird, onethird]))
	numIdenticalConsec = 0
	for i in xrange(numelems-1):
		numIdenticalConsec += (seq[i] == seq[i+1])
	factor(-penaltyMultiplier * numIdenticalConsec)
	numDifferentOpposing = 0
	for i in xrange(numelems/2):
		numDifferentOpposing += (seq[i] != seq[numelems-1-i])
	factor(-penaltyMultiplier * numDifferentOpposing)
	return tuple(seq)

def constrainedStringBTrueDist():
	hist = {}
	for numelems in stringLengths:
		# Probability of choosing this many elements
		numlp = erp.multinomial_logprob(stringLengths.index(numelems), stringLengthProbs)
		for seq in stringsOfLength(numelems, 3):
			# Prior probability of each element value
			lp = -numelems*math.log(3)
			# Identical consecutive element penalty
			numIdenticalConsec = 0
			for i in xrange(numelems-1):
				numIdenticalConsec += (seq[i] == seq[i+1])
			lp -= penaltyMultiplier * numIdenticalConsec
			# Different opposing element penalty
			numDifferentOpposing = 0
			for i in xrange(numelems/2):
				numDifferentOpposing += (seq[i] != seq[numelems-1-i])
			lp -= penaltyMultiplier * numDifferentOpposing
			hist[seq] = numlp + lp
	# Normalize by partition function
	logz = math.log(sum(map(lambda lp: math.exp(lp), hist.values())))
	for seq in hist:
		hist[seq] = math.exp(hist[seq] - logz)
	return hist

def klDivergence(P, Q):
	kldiv = 0.0
	for x in P:
		p = P[x]
		q = Q[x]
		if p != 0.0:
			logq = math.log(q) if q != 0.0 else -float('inf')
			kldiv += (math.log(p) - logq) * p
	return kldiv

def totalVariationDist(P, Q):
	total = 0.0
	for x in P:
		total += abs(P[x] - Q[x])
	return 0.5*total


###############################

if __name__ == "__main__":

	# compareForwardToMHDists(numOnes, 1000)
	# compareForwardToMHDists(sumOfTen, 1000)
	# compareForwardToMHDists(sumOfTenWhile, 1000)
	# compareForwardToMHDists(sumOfTenFor, 1000)
	# compareForwardToMHDists(sumOfTenMap, 1000)
	# compareForwardToMHMeans(oneGaussian, 10000)
	# compareForwardToMHMeans(oneGamma, 10000)
	# compareForwardToMHMeans(oneBeta, 10000)
	# compareForwardToMHMeans(oneBinomial, 10000)
	# compareForwardToMHMeans(onePoisson, 10000)

	# print memTest()

	# print distrib(sprinklerTest, traceMH, 10000)

	# print MAP(oneGaussian, traceMH, 10000)

	# print distrib(constrainedStringA, traceMH, 10000)
	# print "-------------------------------------------"
	# print constrainedStringATrueDist()
	print totalVariationDist(constrainedStringATrueDist(), distrib(constrainedStringA, traceMH, 10000, 1, True))
	print totalVariationDist(constrainedStringATrueDist(), distrib(constrainedStringA, LARJMCMC, 5000, 20, 1, True))
	# print totalVariationDist(constrainedStringBTrueDist(), distrib(constrainedStringB, traceMH, 10000, 1, True))
	# print totalVariationDist(constrainedStringBTrueDist(), distrib(constrainedStringB, LARJMCMC, 10000, 0, 1, True))
	