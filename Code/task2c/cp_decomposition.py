import numpy as np
import cPickle
from tensorly.decomposition import parafac
import util
import os

no_of_components = 5
output_file = 'task2c.out.txt'
output_folder = os.path.join(os.path.dirname(__file__), "..", "..", "Output")

def main():
	actor_movie_year_3d_matrix = cPickle.load( open( "actor_movie_year_tensor.pkl", "rb" ) )
	actor_movie_year_array = np.array(actor_movie_year_3d_matrix)
	decomposed = parafac(actor_movie_year_array, no_of_components)
	# print decomposed[0]
	# print decomposed[1]
	# print decomposed[2]
	#for clearing the output file
	f = open(os.path.abspath(os.path.join(output_folder, output_file)),'w')
	f.close()
	latent_actor_semantics(decomposed[0])
	latent_movie_semantics(decomposed[1])
	latent_year_semantics(decomposed[2])
	partition_components(decomposed)

def latent_actor_semantics(actor_matrix):
	imdb_actor_info = util.read_imdb_actor_info()
	actor_list = imdb_actor_info.id.unique()
	actor_list = imdb_actor_info[imdb_actor_info['id'].isin(actor_list)]['name'].tolist()
	concepts = []
	for i in range(no_of_components):
		concept = []
		for j, component in enumerate(np.transpose(actor_matrix)[i]):
			concept.append((actor_list[j], component))
		concept.sort(key=lambda tup: abs(tup[1]), reverse=True)
		concepts.append(concept)
	util.print_output(concepts, 'Actor')
	util.write_output_file(concepts, output_file, 'Actor')

def latent_movie_semantics(movie_matrix):
	mlmovies = util.read_mlmovies()
	movies_list = mlmovies.movieid.unique()
	movies_list = mlmovies[mlmovies['movieid'].isin(movies_list)]['moviename'].tolist()
	concepts = []
	for i in range(no_of_components):
		concept = []
		for j, component in enumerate(np.transpose(movie_matrix)[i]):
			concept.append((movies_list[j], component))
		concept.sort(key=lambda tup: abs(tup[1]), reverse=True)
		concepts.append(concept)
	util.print_output(concepts, 'Movie')
	util.write_output_file(concepts, output_file, 'Movie')

def latent_year_semantics(year_matrix):
	mlmovies = util.read_mlmovies()
	year_list = mlmovies.year.unique()
	concepts = []
	for i in range(no_of_components):
		concept = []
		for j, component in enumerate(np.transpose(year_matrix)[i]):
			concept.append((year_list[j], component))
		concept.sort(key=lambda tup: abs(tup[1]), reverse=True)
		concepts.append(concept)
	util.print_output(concepts, 'Year')
	util.write_output_file(concepts, output_file, 'Year')

def partition_components(decomposed):
	imdb_actor_info = util.read_imdb_actor_info()
	actor_list = imdb_actor_info.id.unique()
	imdb_actor_info = util.read_imdb_actor_info()
	actor_list = imdb_actor_info[imdb_actor_info['id'].isin(actor_list)]['name'].tolist()
	mlmovies = util.read_mlmovies()
	movies_list = mlmovies.movieid.unique()
	movies_list = mlmovies[mlmovies['movieid'].isin(movies_list)]['moviename'].tolist()
	year_list = mlmovies.year.unique()
	partitions = {1:{'actor': [], 'movie': [], 'year': []}, 
				2:{'actor': [], 'movie': [], 'year': []}, 
				3:{'actor': [], 'movie': [], 'year': []}, 
				4:{'actor': [], 'movie': [], 'year': []}, 
				5:{'actor': [], 'movie': [], 'year': []}
				}
	for j, actor_vec in enumerate(decomposed[0]):
		partition_num = np.argmax(actor_vec)+1
		partitions[partition_num]['actor'].append(actor_list[j])
	for j, movie_vec in enumerate(decomposed[1]):
		partition_num = np.argmax(movie_vec)+1
		partitions[partition_num]['movie'].append(movies_list[j])
	for j, year_vec in enumerate(decomposed[2]):
		partition_num = np.argmax(year_vec)+1
		partitions[partition_num]['year'].append(year_list[j])
	util.print_partition(partitions)
	util.write_partition_output_file(partitions, output_file)

if __name__ == "__main__":
    main()