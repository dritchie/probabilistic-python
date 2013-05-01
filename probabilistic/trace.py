import sys
import copy
from collections import Counter

class RandomVariableRecord:
	"""
	Variables generated by ERPs.
	These form the 'choice points' in a probabilistic program trace.
	"""

	def __init__(self, name, erp, params, val, logprob, structural, conditioned=False):
		self.name = name
		self.erp = erp
		self.params = params
		self.val = val
		self.logprob = logprob
		self.active = True
		self.conditioned = conditioned
		self.structural = structural

class RandomExecutionTrace:
	"""
	Execution trace generated by a probabilistic program.
	Tracks the random choices made and accumulates probabilities
	"""

	def __init__(self, computation, doRejectionInit=True):
		self.computation = computation
		self._vars = {}
		self.varlist = []
		self.currVarIndex = 0
		self.logprob = 0
		self.newlogprob = 0		# From newly-added variables
		self.oldlogprob = 0		# From unreachable variables
		self.rootframe = None
		self.loopcounters = Counter()
		self.conditionsSatisfied = False
		self.returnValue = None
		if doRejectionInit:
			while not self.conditionsSatisfied:
				self._vars.clear()
				self.traceUpdate()

	def __deepcopy__(self, memo):
		newdb = RandomExecutionTrace(self.computation, doRejectionInit=False)
		newdb.logprob = self.logprob
		newdb.oldlogprob = self.oldlogprob
		newdb.newlogprob = self.newlogprob
		newdb.varlist = [copy.copy(record) for record in self.varlist]
		newdb._vars = {record.name:record for record in newdb.varlist}
		newdb.conditionsSatisfied = self.conditionsSatisfied
		newdb.returnValue = self.returnValue
		return newdb

	def freeVarNames(self, structural=True, nonstructural=True):
		return map(lambda tup: tup[0], \
				   filter(lambda tup: not tup[1].conditioned and \
				   					  ((structural and tup[1].structural) or (nonstructural and not tup[1].structural)), \
						  self._vars.iteritems()))

	def varDiff(self, other):
		"""
		The names of the variables that this trace has that the other trace does not have
		"""
		return list(set(self._vars.keys()) - set(other._vars.keys()))

	def lpDiff(self, other):
		"""
		The difference in log probability between this trace and the other resulting
		from the variables that this has that the other does not
		"""
		return sum(map(lambda name: self._vars[name].logprob, self.varDiff(other)))

	def traceUpdate(self, structureIsFixed=False):
		"""
		Run computation and update this trace accordingly
		"""

		global _trace
		originalTrace = _trace
		_trace = self

		self.logprob = 0.0
		self.newlogprob = 0.0
		self.loopcounters.clear()
		self.conditionsSatisfied = True
		self.currVarIndex = 0

		# If updating this trace can change the variable structure, then we
		# clear out the flat list of variables beforehand
		if not structureIsFixed:
			self.varlist = []

		# First, mark all random values as 'inactive'; only
		# those reeached by the computation will become 'active'
		for record in self._vars.values():
			record.active = False

		# Mark that this is the 'root' of the current execution trace
		self.rootframe = sys._getframe()

		# Run the computation, which will create/lookup random variables
		self.returnValue = self.computation()

		# Clear out the root frame, etc.
		self.rootframe = None
		self.loopcounters.clear()

		# Clean up any random values that are no longer reachable
		self.oldlogprob = 0.0
		for record in self._vars.values():
			if not record.active:
				self.oldlogprob += record.logprob
		self._vars = {name:record for name,record in self._vars.iteritems() if record.active}

		_trace = originalTrace

	def proposeChange(self, varname, structureIsFixed=False):
		"""
		Propose a random change to the variable name 'varname'
		Returns a new sample trace from the computation and the
			forward and reverse probabilities of proposing this change
		"""
		nextTrace = copy.deepcopy(self)
		var = nextTrace.getRecord(varname)
		propval = var.erp._proposal(var.val, var.params)
		fwdPropLP = var.erp._logProposalProb(var.val, propval, var.params)
		rvsPropLP = var.erp._logProposalProb(propval, var.val, var.params)
		var.val = propval
		var.logprob = var.erp._logprob(var.val, var.params)
		nextTrace.traceUpdate(structureIsFixed)
		fwdPropLP += nextTrace.newlogprob
		rvsPropLP += nextTrace.oldlogprob
		return nextTrace, fwdPropLP, rvsPropLP

	def currentName(self, numFrameSkip):
		"""
		Return the current name, as determined by the interpreter
			stack of the current program.
		Skips the top 'numFrameSkip' stack frames that precede this
			function's stack frame (numFrameSkip+1 frames total)
		"""

		# Get list of frames from the root to the current frame
		f = sys._getframe(numFrameSkip+1)
		flst = []
		while f and f is not self.rootframe:
			flst.insert(0, f)
			f = f.f_back

		# Build up name string, checking loop counters along the way
		name = ""
		for i in xrange(len(flst)-1):
			f = flst[i]
			name += "{0}:{1}".format(id(f.f_code), f.f_lasti)
			loopnum = self.loopcounters[name]
			name += ":{0}|".format(loopnum)
		# For the last (topmost) frame, also increment the loop counter
		f = flst[-1]
		name += "{0}:{1}".format(id(f.f_code), f.f_lasti)
		loopnum = self.loopcounters[name]
		self.loopcounters[name] += 1
		name += ":{0}|".format(loopnum)

		return name

	def lookup(self, erp, params, numFrameSkip, isStructural, conditionedValue=None):
		"""
		Looks up the value of a random variable.
		If this random variable does not exist, create it
		"""

		record = None
		name = None
		# Try to find the variable (first check the flat list, then do
		# slower structural name lookup)
		varIsInFlatList = self.currVarIndex < len(self.varlist)
		if varIsInFlatList:
			record = self.varlist[self.currVarIndex]
		else:
			name = self.currentName(numFrameSkip+1)
			record = self._vars.get(name)
			if (not record or record.erp is not erp or isStructural != record.structural):
				record = None
		# If we didn't find the variable, create a new one
		if not record:
			val = (conditionedValue if conditionedValue else erp._sample_impl(params))
			ll = erp._logprob(val, params)
			self.newlogprob += ll
			record = RandomVariableRecord(name, erp, params, val, ll, isStructural, conditionedValue != None)
			self._vars[name] = record
		# Otherwise, reuse the variable we found, but check if its parameters/conditioning
		# status have changed
		else:
			record.conditioned = (conditionedValue != None)
			hasChanges = False
			if record.params != params:
				record.params = params
				hasChanges = True
			if conditionedValue and conditionedValue != record.val:
				record.val = conditionedValue
				record.conditioned = True
				hasChanges = True
			if hasChanges:
				record.logprob = erp._logprob(record.val, record.params)

		# Finish up and return
		if not varIsInFlatList:
			self.varlist.append(record)
		self.currVarIndex += 1
		self.logprob += record.logprob
		record.active = True
		return record.val

	def getRecord(self, name):
		"""
		Simply retrieve the variable record associated with name
		"""
		return self._vars.get(name)

	def addFactor(self, num):
		"""
		Add a new factor into the log likelihood of the current trace
		"""
		self.logprob += num

	def conditionOn(self, boolexpr):
		"""
		Condition the trace on the value of a boolean expression
		"""
		self.conditionsSatisfied = self.conditionsSatisfied and boolexpr

"""
Global singleton instance
"""
_trace = None

def lookupVariableValue(erp, params, isStructural, numFrameSkip, conditionedValue=None):
	global _trace
	if not _trace:
		return (conditionedValue if conditionedValue else erp._sample_impl(params))
	else:
		return _trace.lookup(erp, params, numFrameSkip+1, isStructural, conditionedValue)

def newTrace(computation):
	return RandomExecutionTrace(computation)

def factor(num):
	global _trace
	if _trace:
		_trace.addFactor(num)

def condition(boolexpr):
	global _trace
	if _trace:
		_trace.conditionOn(boolexpr)