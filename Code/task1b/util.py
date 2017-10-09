from datetime import datetime
import pandas as pd

db_folder = '../phase2_data/'
mltags_file = 'mltags.csv'
mlmovies_file = 'mlmovies.csv'
genome_tags_file = 'genome-tags.csv'


############### HELPER FUNCTION TO READ FILES #################################

# import mltags
def read_mltags():
	mltags =  pd.read_csv(db_folder + mltags_file)	
	current_time = datetime.now()
	for i,row in mltags.iterrows():
		mltags.set_value(i,'timestamp', (datetime.strptime(row['timestamp'],'%Y-%m-%d %H:%M:%S') - datetime.fromtimestamp(0)).total_seconds()/
			(current_time - datetime.fromtimestamp(0)).total_seconds())
	return mltags

# import mlmovies
def read_mlmovies():
	return pd.read_csv(db_folder + mlmovies_file)

# import genome-tags
def read_genome_tags():
	return pd.read_csv(db_folder + genome_tags_file)

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
	genre_movie_count = movieids_of_genre.shape[0]

	return (genre_movie_count, mltags_of_genre)

def get_tf_idf_matrix(genre):
	mltags = get_mltags()
	mlmovies = get_mlmovies()
	movie_genre = get_movie_genre(mlmovies)
	genre_movie_count, mltags_of_genre = filter_by_genre(mltags, mlmovies, genre)

	# Needed for TF denominators
	tags_per_movie = mltags_of_genre.groupby('movieid', as_index=False)['tagid'].agg({'m_count' : 'count'})

	# Needed for IDF denonminators
	movies_per_tag = mltags_of_genre.groupby('tagid', as_index=False)['movieid'].agg({'t_count' : pd.Series.nunique})

	tagid_movieid_grouped = mltags_of_genre.groupby(['tagid', 'movieid'], as_index=False).agg({'timestamp': 'sum'})

	merged1 = pd.merge(tagid_movieid_grouped, tags_per_movie, on=['movieid','movieid'], how='inner')
	merged2 = pd.merge(merged1, movies_per_tag, on=['tagid', 'tagid'], how = 'inner')
	print merged2
	#tbl2 = tbl.pivot(index='movieid', columns='tagid', values='timestamp').fillna(0)
	#print tbl2

