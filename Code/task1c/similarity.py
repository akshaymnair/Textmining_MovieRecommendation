import pandas as pd
import sys
import util
import os
from scipy.spatial.distance import cosine

output_file = 'task1c_similarity.out.txt'
no_of_actors = 10

def main():
	if len(sys.argv) < 2:
		print('Expected arguments are not provided.')
		return
	actorid = int(sys.argv[1])
	imdb_actor_info = util.read_imdb_actor_info()
	input_actor = imdb_actor_info[imdb_actor_info['id'] == actorid]['name'].values[0]

	tf_idf_matrix = util.get_tf_idf_matrix()
	#print (tf_idf_matrix)
	input_actor_tf_idf = tf_idf_matrix.loc[actorid]
	#print (input_actor_tf_idf)

	actors = []
	for index, row in tf_idf_matrix.iterrows():
		actor_name = imdb_actor_info[imdb_actor_info['id'] == index]['name'].values[0]
		actors.append((index, actor_name, 1 - cosine(row, input_actor_tf_idf)))
	other_actors = list(filter(lambda tup: tup[0] != actorid, actors))
	other_actors.sort(key=lambda tup: tup[2], reverse=True)
	
	util.print_output(actorid, input_actor, other_actors[:no_of_actors])
	util.write_output_file(actorid, input_actor, other_actors[:no_of_actors], output_file)

if __name__ == "__main__":
    main()
