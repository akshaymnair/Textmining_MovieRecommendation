import pandas as pd
import sys
import util
import os
from scipy.spatial.distance import cosine

output_file = 'task1c_similarity.out.txt'

def main():
	if len(sys.argv) < 2:
		print('Expected arguments are not provided.')
		return
	actorid = sys.argv[1]

	tf_idf_matrix = util.get_tf_idf_matrix()
	print(tf_idf_matrix)
	#actor_tf_idf = tf_idf_matrix.where(tf_idf_matrix['actorid']==actorid).dropna()
	#print (actor_tf_idf)

	#cos_sim = cosine_similarity(tf_idf_matrix)
	#print (cos_sim)

if __name__ == "__main__":
    main()
