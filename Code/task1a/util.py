from datetime import datetime
import pandas as pd
from numpy import log
import os

db_folder_path = os.path.join(os.path.dirname(__file__), "..", "..", "Phase2_data")
output_folder = os.path.join(os.path.dirname(__file__), "..", "..", "Output")

mltags_file = 'mltags.csv'
mlmovies_file = 'mlmovies.csv'
genome_tags_file = 'genome-tags.csv'


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

################## HELPER FUNCTION TO PROCESS AND RETRIEVE ######################

def get_movie_genre(mlmovies):
	movie_genre = pd.DataFrame(mlmovies.genres.str.split('|').tolist(), index = mlmovies.movieid).stack()
	movie_genre = movie_genre.reset_index()[[0,'movieid']]
	movie_genre.columns = ['genre','movieid']
	return movie_genre

def filter_by_genre(mltags, mlmovies, genre):
	movie_genre = get_movie_genre(mlmovies)
	movieids_of_genre = movie_genre.where(movie_genre['genre']==genre).dropna().loc[:,'movieid'].unique()
	mltags_of_genre = mltags.where(mltags['movieid'].isin(movieids_of_genre)).dropna()
	genre_movie_count = int(movieids_of_genre.shape[0])

	return (genre_movie_count, mltags_of_genre)

def get_tf_idf_matrix(genre):
	mltags = read_mltags()
	mlmovies = read_mlmovies()
	movie_genre = get_movie_genre(mlmovies)
	genre_movie_count, mltags_of_genre = filter_by_genre(mltags, mlmovies, genre)

	# Needed for TF denominators. Calcuate total number of tags per movie
	tags_per_movie = mltags_of_genre.groupby('movieid', as_index=False)['tagid'].agg({'m_count' : pd.Series.count})

	# Needed for IDF denonminators. Calculate unique movieids for each tag
	movies_per_tag = mltags_of_genre.groupby('tagid', as_index=False)['movieid'].agg({'t_count' : pd.Series.nunique})

	# Grouped so as to create unique tagid, movieid pairs and calculate term frequency from timestamp
	tagid_movieid_grouped = mltags_of_genre.groupby(['tagid', 'movieid'], as_index=False)['timestamp'].agg({'tf': 'sum'})

	# Merge tag_counts. Add new column including calculated tags_per_movie.
	M1 = pd.merge(tagid_movieid_grouped, tags_per_movie, on=['movieid','movieid'], how='inner')

	# Merge movie_counts. Add new column including calculated movies_per_tag.
	M2 = pd.merge(M1, movies_per_tag, on=['tagid', 'tagid'], how = 'inner')

	# Perform TF-IDF from the data.
	M2['tfidf'] = M2['tf']*log(genre_movie_count/M2['m_count'])/M2['t_count']
	#print M2

	# Pivot the matrix to get in required form 
	R = M2.pivot(index='movieid', columns='tagid', values='tfidf').fillna(0)
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

