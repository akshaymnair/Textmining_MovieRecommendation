import pandas as pd
import numpy as np
import cPickle
from tensorly.decomposition import parafac
import util
import os

no_of_components = 5
output_file = 'task2d.out.txt'
output_folder = os.path.join(os.path.dirname(__file__), "..", "..", "Output")

def main():
	tag_movie_rating_3d_matrix = cPickle.load( open( "tag_movie_rating_tensor.pkl", "rb" ) )
	tag_movie_rating_array = np.array(tag_movie_rating_3d_matrix)
	decomposed = parafac(tag_movie_rating_array, no_of_components, init='random')
	# print decomposed[0]
	# print decomposed[1]
	# print decomposed[2]
	#for clearing the output file
	f = open(os.path.abspath(os.path.join(output_folder, output_file)),'w')
	f.close()
	latent_tag_semantics(decomposed[0])
	latent_movie_semantics(decomposed[1])
	latent_rating_semantics(decomposed[2])
	partition_components(decomposed)

def latent_tag_semantics(tag_matrix):
	mltags = util.read_mltags()
	genome_tags = util.read_genome_tags()
	mltags = pd.merge(mltags, genome_tags, left_on='tagid', right_on='tagId', how='inner')
	tags_list = mltags.tagid.unique()
	tags_list = mltags[mltags['tagid'].isin(tags_list)]['tag'].tolist()
	concepts = []
	for i in range(no_of_components):
		concept = []
		for j, component in enumerate(np.transpose(tag_matrix)[i]):
			concept.append((tags_list[j], component))
		concept.sort(key=lambda tup: abs(tup[1]), reverse=True)
		concepts.append(concept)
	util.print_output(concepts, 'Tag')
	util.write_output_file(concepts, output_file, 'Tag')

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

def latent_rating_semantics(rating_matrix):
	mlratings = util.read_mlratings()
	ratings_list = mlratings.rating.unique()
	concepts = []
	for i in range(no_of_components):
		concept = []
		for j, component in enumerate(np.transpose(rating_matrix)[i]):
			concept.append((ratings_list[j], component))
		concept.sort(key=lambda tup: abs(tup[1]), reverse=True)
		concepts.append(concept)
	util.print_output(concepts, 'Rating')
	util.write_output_file(concepts, output_file, 'Rating')

def partition_components(decomposed):
	mltags = util.read_mltags()
	genome_tags = util.read_genome_tags()
	mltags = pd.merge(mltags, genome_tags, left_on='tagid', right_on='tagId', how='inner')
	tags_list = mltags.tagid.unique()
	tags_list = genome_tags[genome_tags['tagId'].isin(tags_list)]['tag'].tolist()
	mlmovies = util.read_mlmovies()
	movies_list = mlmovies.movieid.unique()
	movies_list = mlmovies[mlmovies['movieid'].isin(movies_list)]['moviename'].tolist()
	mlratings = util.read_mlratings()
	ratings_list = mlratings.rating.unique()
	partitions = {1:{'tag': [], 'movie': [], 'rating': []}, 
				2:{'tag': [], 'movie': [], 'rating': []}, 
				3:{'tag': [], 'movie': [], 'rating': []}, 
				4:{'tag': [], 'movie': [], 'rating': []}, 
				5:{'tag': [], 'movie': [], 'rating': []}
				}
	for j, tag_vec in enumerate(decomposed[0]):
		partition_num = np.argmax(tag_vec)+1
		partitions[partition_num]['tag'].append(tags_list[j])
	for j, movie_vec in enumerate(decomposed[1]):
		partition_num = np.argmax(movie_vec)+1
		partitions[partition_num]['movie'].append(movies_list[j])
	for j, rating_vec in enumerate(decomposed[2]):
		partition_num = np.argmax(rating_vec)+1
		partitions[partition_num]['rating'].append(ratings_list[j])
	util.print_partition(partitions)
	util.write_partition_output_file(partitions, output_file)

if __name__ == "__main__":
    main()