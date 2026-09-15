def vector_sum(a: list[int|float], b: list[int|float]) -> list[int|float]:
	# Return the element-wise sum of vectors 'a' and 'b'.
	# If vectors have different lengths, return -1.
	value=[]
	if not len(a) == len(b):
		return [-1]
	for i in range(len(a)):
		# if (isinstance(a[i],list)) or (isinstance(b[i],list)):
		s = a[i]+b[i]
		value.append(s)
	return value
	pass