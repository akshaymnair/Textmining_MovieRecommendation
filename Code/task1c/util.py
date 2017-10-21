from datetime import datetime
import pandas as pd
from numpy import log
import os

db_folder_path = os.path.join(os.path.dirname(__file__), "..", "..", "Phase2_data")
output_folder = os.path.join(os.path.dirname(__file__), "..", "..", "Output")

mltags_file = 'mltags.csv'
mlmovies_file = 'mlmovies.csv'
genome_tags_file = 'genome-tags.csv'
movie_actor_file = 'movie-actor.csv'

############### HELPER FUNCTION TO READ FILES #################################

# import mltags
def read_mltags():
	mltags =  pd.read_csv(os.path.abspath(os.path.join(db_folder_path, mltags_file)))
	current_time = datetime.now()
	for i,row in mltags.iterrows():
		mltags.set_value(i,'timestamp', (datetime.strptime(row['timestamp'],'%Y-%m-%d %H:%M:%S') - datetime.fromtimestamp(0)).total_seconds()/
			(current_time - datetime.fromtimestamp(0)).total_seconds())
	return mltags

# import mlmovies
def read_mlmovies():
	return pd.read_csv(os.path.abspath(os.path.join(db_folder_path, mlmovies_file)))

# import genome-tags
def read_genome_tags():
	return pd.read_csv(os.path.abspath(os.path.join(db_folder_path, genome_tags_file)))

def read_movie_actor():
	return pd.read_csv(os.path.abspath(os.path.join(db_folder_path, movie_actor_file)))

################## HELPER FUNCTION TO PROCESS AND RETRIEVE ######################

def get_movie_genre(mlmovies):
	movie_genre = pd.DataFrame(mlmovies.genres.str.split('|').tolist(), index = mlmovies.movieid).stack()
	movie_genre = movie_genre.reset_index()[[0,'movieid']]
	movie_genre.columns = ['genre','movieid']
	return movie_genre

def filter_by_actor(mltags, mlmovies, movie_actor):
	movie_genre = get_movie_genre(mlmovies)
	movieids_of_genre = movie_genre.where(movie_genre['genre']==genre).dropna().loc[:,'movieid'].unique()
	mltags_of_actor = mltags.where(mltags['movieid'].isin(movieids_of_genre)).dropna()
	genre_movie_count = int(movieids_of_genre.shape[0])
	return (genre_movie_count, mltags_of_actor)

def get_tf_idf_matrix():
	mltags = read_mltags()
	movie_actor = read_movie_actor()

	# Needed for IDF Numerator. Calculate total movies
	mltags_with_actor = pd.merge(movie_actor, mltags, on='movieid')
	actor_count = int(mltags_with_actor.loc[:, 'actorid'].unique().shape[0])

	# Needed for TF denominators. Calcuate total number of tags per movie
	tags_per_actor = mltags_with_actor.groupby('actorid', as_index=False)['tagid'].agg({'t_count' : pd.Series.count})

	# Needed for IDF denonminators. Calculate unique movieids for each tag
	actors_per_tag = mltags_with_actor.groupby('tagid', as_index=False)['actorid'].agg({'a_count' : pd.Series.nunique})

	#term is created to merge actor_movie_rank and timestamp
	mltags_with_actor['term'] = mltags_with_actor['timestamp']*(100/(99+mltags_with_actor['actor_movie_rank']))

	# Grouped so as to create unique tagid, movieid pairs and calculate term frequency from timestamp
	actorid_tagid_grouped = mltags_with_actor.groupby(['tagid', 'actorid'], as_index=False)['term'].agg({'tf': 'sum'})

	# Merge tag_counts. Add new column including calculated tags_per_movie.
	M1 = pd.merge(actorid_tagid_grouped, tags_per_actor, on=['actorid','actorid'], how='inner')

	# Merge movie_counts. Add new column including calculated movies_per_tag.
	M2 = pd.merge(M1, actors_per_tag, on=['tagid', 'tagid'], how = 'inner')

	# Perform TF-IDF from the data.
	M2['tfidf'] = M2['tf']*log(actor_count/M2['a_count'])/M2['t_count']
	#print M2

	# Pivot the matrix to get in required form 
	R = M2.pivot(index='actorid', columns='tagid', values='tfidf').fillna(0)
	#print R
	return R

def print_output(genre, concepts):
	print('For genre: ' + genre + ', output is :')
	for idx, concept in enumerate(concepts):
		print ('\nThe ' + str(idx+1) +'th concept is: ')
		print('%40s\t%15s\t' %('Tag', 'Weight'))
		for row in concept:
			print ('%40s\t%15s\t' %(row[0], row[1]))

def write_output_file(genre, concepts, filename):
	f = open(os.path.abspath(os.path.join(output_folder, filename)),'w')
	f.write('For genre: ' + genre + ', output is :' + '\n')
	for idx, concept in enumerate(concepts):
		f.write('\nThe ' + str(idx+1) +'th concept is: ')
		f.write('\n%40s\t%15s\t' %('Tag', 'Weight'))
		for row in concept:
			f.write('\n%40s\t%15s\t' %(row[0], row[1]))

