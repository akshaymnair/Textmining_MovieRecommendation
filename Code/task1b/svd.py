import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import sparse_random_matrix
import sys
import util
import os

output_file = 'task1b_svd.out.txt'

def main():
	if len(sys.argv) < 2:
		print('Expected arguments are not provided.')
		return
	genre = sys.argv[1]
	no_of_components = 4
	imdb_actor_info = util.read_imdb_actor_info()
	#print imdb_actor_info

	tf_idf_matrix = util.get_tf_idf_matrix(genre)
	actor_list = list(tf_idf_matrix.columns.values)
	actor_list = imdb_actor_info[imdb_actor_info['id'].isin(actor_list)]['name'].tolist()
	#print actor_list
	
	svd = TruncatedSVD(n_components=no_of_components, n_iter=100, random_state=None)
	svd.fit(tf_idf_matrix)

	concepts = []
	for i in range(no_of_components):
		concept = []
		for j, component in enumerate(svd.components_[i]):
			concept.append((actor_list[j], component))
		concept.sort(key=lambda tup: abs(tup[1]), reverse=True)
		concepts.append(concept)
	util.print_output(genre, concepts)
	util.write_output_file(genre, concepts, output_file)

if __name__ == "__main__":
    main()
