from probabilistic import *
import probabilistic.database
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
	database.getCurrentDatabase().recording = False
	while i < iters:
		i += 1
		hist[len(ones())] += 1
	database.getCurrentDatabase().recording = True
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

if __name__ == "__main__":
	forwardhist = onesDistribForward(1000)
	print "Foward hist:"
	print forwardhist
	#print "Forward: Average seq length: {0}".format(forwardavglen)
	mhhist = onesDistribMH(1000)
	print "MH hist:"
	print mhhist
	#mhavglen = sum(map(lambda seq: len(seq), mhsamps)) / float(len(mhsamps))
	#print "MH: Average seq length: {0}".format(mhavglen)
