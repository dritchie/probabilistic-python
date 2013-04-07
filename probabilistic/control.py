import trace

def ntimes(times, block):
	"""
	Repeat a computation n times
	"""
	for i in xrange(times):
		trace.incrementLoopCounter(0)
		block(i)

def foreach(iterable, block):
	"""
	'for' loop control structure suitable for use inside probabilistic programs.
	Invokes block for every element in iterable.
	"""
	for elem in iterable:
		trace.incrementLoopCounter(0)
		block(elem)

def until(condition, block):
	"""
	'while' loop control structure suitable for use inside probabilistic programs.
	Invokes block until condition is true.
	"""
	cond = condition()
	while not cond:
		trace.incrementLoopCounter(0)
		block()
		cond = condition()

_map = map
def map(proc, iterable):
	"""
	Higher-order 'map' function suitable for use inside probabilistic programs.
	Transforms every element of iterable using proc, returning a new sequence object.
	"""
	def procwrapper(elem):
		trace.incrementLoopCounter(1)
		return proc(elem)
	return _map(procwrapper, iterable)

def repeat(times, proc):
	"""
	Evaluate proc() 'times' times and build a list out of the results
	"""
	return map(lambda x: proc(), range(times))