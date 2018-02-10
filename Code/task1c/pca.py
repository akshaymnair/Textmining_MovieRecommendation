import pandas as pd
import sys
import util
import os
from scipy.spatial.distance import cosine
from sklearn.decomposition import PCA

output_file = 'task1c_pca.out.txt'

no_of_actors = 10
no_of_components = 5

def main():
	if len(sys.argv) < 2:
		print('Expected arguments are not provided.')
		return
	actorid = int(sys.argv[1])
	imdb_actor_info = util.read_imdb_actor_info()
	input_actor_name = imdb_actor_info[imdb_actor_info['id'] == actorid]['name'].values[0]

	tf_idf_matrix = util.get_tf_idf_matrix()
	#print(tf_idf_matrix)	
	actor_tf_idf = tf_idf_matrix.loc[actorid]
	#print(actor_tf_idf)

	pca = PCA(n_components=no_of_components)
	pca.fit(tf_idf_matrix)
	pca_df = pd.DataFrame(pca.transform(tf_idf_matrix), index=tf_idf_matrix.index)

	input_actor_row = pca_df.loc[actorid]

	actors = []
	for index, row in pca_df.iterrows():
		name = imdb_actor_info[imdb_actor_info['id'] == index]['name'].values[0]
		actors.append((index, name, 1 - cosine(row, input_actor_row)))
	other_actors = list(filter(lambda tup: tup[0] != actorid, actors))
	other_actors.sort(key=lambda tup: tup[2], reverse=True)
	util.print_output(actorid, input_actor_name, other_actors[:no_of_actors])
	util.write_output_file(actorid, input_actor_name, other_actors[:no_of_actors], output_file)

if __name__ == "__main__":
    main()
