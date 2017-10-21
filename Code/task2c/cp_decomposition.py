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

def latent_actor_semantics(actor_matrix):
	imdb_actor_info = util.read_imdb_actor_info()
	actor_list = imdb_actor_info.id.unique()
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

if __name__ == "__main__":
    main()