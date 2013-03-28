from probabilistic import *
import math
from collections import Counter

def ones():
	if flip(0.75):
		return [1] + ones()
	else:
		return []

def onesDistribForward(iters):
	hist = Counter()
	i = 0
	while i < iters:
		i += 1
		hist[len(ones())] += 1
	return hist

def onesDistribMH(iters):
	hist = Counter()
	samps = sample(ones, 1000)
	for s in samps:
		hist[len(s)] += 1
	return hist

def constrainedOnes():
	seq = ones()
	factor(-math.pow(abs(len(seq) - 4), 10))
	return seq

def sumOfTenFlips():
	num = 0
	num += flip(0.5)
	num += flip(0.5)
	num += flip(0.5)
	num += flip(0.5)
	num += flip(0.5)
	num += flip(0.5)
	num += flip(0.5)
	num += flip(0.5)
	num += flip(0.5)
	num += flip(0.5)
	return num

def tenFlipsDistribForward(iters):
	hist = Counter()
	i = 0
	while i < iters:
		i += 1
		hist[sumOfTenFlips()] += 1
	return hist

def tenFlipsDistribMH(iters):
	hist = Counter()
	samps = sample(sumOfTenFlips, iters)
	#samps = sample(sumOfTenFlips, iters, lambda n: n >= 5)
	for s in samps:
		hist[s] += 1
	return hist

if __name__ == "__main__":
	# forwardhist = onesDistribForward(1000)
	# print "Foward hist:"
	# print forwardhist
	# mhhist = onesDistribMH(1000)
	# print "MH hist:"
	# print mhhist
	#################
	# forwardhist = tenFlipsDistribForward(1000)
	# print "Foward hist:"
	# print forwardhist
	# mhhist = tenFlipsDistribMH(1000)
	# print "MH hist:"
	# print mhhist
	#################
	# avglen = sum(map(lambda s: len(s), sample(constrainedOnes, 1000))) / 1000.0
	# print "Average length: {0}".format(avglen)
