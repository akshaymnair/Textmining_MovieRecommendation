import pandas as pd
import sys
import util
import os
from sklearn.decomposition import LatentDirichletAllocation as LDA

no_of_actors = 10
no_of_components = 5

output_file = 'task1d_lda.out.txt'

def main():
	if len(sys.argv) < 2:
		print('Expected arguments are not provided.')
		return
	movieid = int(sys.argv[1])
	mlmovies = util.read_mlmovies()
	movie_actors = util.read_movie_actor()
	imdb_actor_info = util.read_imdb_actor_info()

	input_movie = mlmovies[mlmovies['movieid'] == movieid]['moviename'].values[0]
	actors_of_movie = movie_actors.where(movie_actors['movieid']==movieid).dropna().loc[:,'actorid'].unique()
	#print (actors_of_movie)

	movie_matrix = util.get_movie_tf_matrix()
	actor_matrix = util.get_actor_tf_matrix()
	#print(actor_matrix.shape)
	input_movie_vector = pd.DataFrame(movie_matrix.loc[movieid])#.transpose()
	#print(input_movie_vector.shape)
	#similarity_matrix = actor_matrix.dot(input_movie_vector)
	#similarity_matrix = similarity_matrix[~similarity_matrix.index.isin(actors_of_movie)]
	#print(similarity_matrix)

	lda = LDA(n_components=no_of_components)
	lda.fit(movie_matrix)
	lda_movie_df = pd.DataFrame(lda.transform(movie_matrix), index=movie_matrix.index)

	input_movie_vector = pd.DataFrame(lda_movie_df.loc[movieid])
	#print(input_movie_vector)


	df_components = pd.DataFrame(lda.components_, columns=actor_matrix.columns).transpose()
	#print(df_components.shape)
	#print(df_components)
	#print(actor_matrix.shape)
	projected_matrix = actor_matrix.dot(df_components)
	#print(projected_matrix)

	similarity_matrix = projected_matrix.dot(input_movie_vector)
	similarity_matrix = similarity_matrix[~similarity_matrix.index.isin(actors_of_movie)]
	#print(similarity_matrix)

	actors = []
	for index, row in similarity_matrix.iterrows():
		actor_name = imdb_actor_info[imdb_actor_info['id'] == index]['name'].values[0]
		actors.append((index, actor_name, similarity_matrix.loc[index][movieid]))
	actors.sort(key=lambda tup: tup[2], reverse=True)
	#print (actors)
	
	util.print_output(movieid, input_movie, actors[:no_of_actors])
	util.write_output_file(movieid, input_movie, actors[:no_of_actors], output_file)

if __name__ == "__main__":
    main()
