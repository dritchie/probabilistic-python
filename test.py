from probabilistic import *
import probabilistic.database
import math

def ones():
	if flip(0.75):
		return [1] + ones()
	else:
		return []

def constrainedOnes():
	seq = ones()
	#factor(-math.pow(abs(len(seq) - 4), 2))
	return seq

if __name__ == "__main__":
	database.getCurrentDatabase().recording = False
	forwardsamps = [ones() for i in range(1000)]
	database.getCurrentDatabase().recording = True
	forwardavglen = sum(map(lambda seq: len(seq), forwardsamps)) / float(len(forwardsamps))
	print "Forward: Average seq length: {0}".format(forwardavglen)
	mhsamps = sample(ones, 1000)
	mhavglen = sum(map(lambda seq: len(seq), mhsamps)) / float(len(mhsamps))
	print "MH: Average seq length: {0}".format(mhavglen)
