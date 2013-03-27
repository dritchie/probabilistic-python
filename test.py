from probabilistic import *
import math

def ones():
	if flip(0.75):
		return [1] + ones()
	else:
		return []

def constrainedOnes():
	seq = ones()
	factor(-math.pow(abs(len(seq) - 4), 2))
	return seq

if __name__ == "__main__":
	#print ones()
	samps = sample(constrainedOnes, 1000)
	for seq in samps[990:1000]:
		print seq
