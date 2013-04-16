import trace

def ntimes(times, block):
	"""
	Repeat a computation n times
	"""
	for i in xrange(times):
		block(i)

def foreach(iterable, block):
	"""
	'for' loop control structure suitable for use inside probabilistic programs.
	Invokes block for every element in iterable.
	"""
	for elem in iterable:
		block(elem)

def until(condition, block):
	"""
	'while' loop control structure suitable for use inside probabilistic programs.
	Invokes block until condition is true.
	"""
	cond = condition()
	while not cond:
		block()
		cond = condition()

def repeat(times, proc):
	"""
	Evaluate proc() 'times' times and build a list out of the results
	"""
	return map(lambda x: proc(), range(times))