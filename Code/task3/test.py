import pandas as pd
import actor_vector as av
import numpy as np
import math

def cosine_similarity(list1,list2):
	dic1 = {k[0]:k[1] for k in list1}
	dic2 = {k[0]:k[1] for k in list2}
	numerator = 0.0
	dena = 0.0
	for key1,val1 in dic1.iteritems():
		numerator += val1*dic2.get(key1,0.0)
		dena += val1*val1
	denb = 0.0
	for val2 in dic2.values():
		denb += val2*val2
	if (dena!=0.0) & (denb!= 0.0):   
		print numerator/math.sqrt(dena*denb) 
		return numerator/math.sqrt(dena*denb)
	return 0.0


cosine_similarity(av.tfidf(1792455),av.tfidf(396877))
