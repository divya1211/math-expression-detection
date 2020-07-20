
def huristicsPass(inputString, alphaThreshold=0.9, maxInputLen=0):
	try:
		alphaRatio = sum(not char.isdigit() for char in inputString)/len(inputString)
		return alphaRatio < alphaThreshold and len(inputString) > maxInputLen
	except:
		return False
