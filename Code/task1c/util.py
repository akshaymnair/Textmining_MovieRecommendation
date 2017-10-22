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
imdb_actor_info_file = 'imdb-actor-info.csv'

############### HELPER FUNCTION TO READ FILES #################################

# import mltags
def read_mltags():
	mltags =  pd.read_csv(os.path.abspath(os.path.join(db_folder_path, mltags_file)))
	current_time = datetime.now()
	for i,row in mltags.iterrows():
		mltags.set_value(i,'timestamp', (datetime.strptime(row['timestamp'],'%Y-%m-%d %H:%M:%S') - datetime.fromtimestamp(0)).total_seconds()/
			(current_time - datetime.fromtimestamp(0)).total_seconds())
	return mltags

# import imdb-actor-info
def read_imdb_actor_info():
	return pd.read_csv(os.path.abspath(os.path.join(db_folder_path, imdb_actor_info_file)))

def read_movie_actor():
	return pd.read_csv(os.path.abspath(os.path.join(db_folder_path, movie_actor_file)))

################## HELPER FUNCTION TO PROCESS AND RETRIEVE ######################

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
	R = M2.pivot_table(index='actorid', columns='tagid', values='tfidf').fillna(0)
	#print R
	return R

def print_output(actorid, actor, actors):
	print('For actorid: ' + str(actorid) + ' and actor: '+ actor + ', output is :')
	print('\n%15s\t%40s\t%15s\t' %('Actorid', 'Actor name', 'cosine similarity'))
	for idx, row in enumerate(actors):
		print ('%15s\t%40s\t%15s\t' %(str(row[0]), row[1], str(row[2])))

def write_output_file(actorid, actor, actors, filename):
	f = open(os.path.abspath(os.path.join(output_folder, filename)),'w')
	f.write('For actorid: ' + str(actorid) + ' and actor: '+ actor + ', output is :'+ '\n')
	f.write('\n%15s\t%40s\t%15s\t' %('Actorid', 'Actor name', 'cosine similarity'))
	for idx, row in enumerate(actors):
		f.write('\n%15s\t%40s\t%15s\t' %(row[0], row[1], row[2]))

