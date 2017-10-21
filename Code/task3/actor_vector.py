from __future__ import division
import pandas as pd
from datetime import datetime
from math import log
from operator import itemgetter
import sys

def tfidf(actor_id):
	current_time = datetime.now()
	tags = pd.read_csv('../../Phase2_data/mltags.csv')
	movie_actor = pd.read_csv('../../Phase2_data/movie-actor.csv')
	tags_details = pd.read_csv('../../Phase2_data/genome-tags.csv')
	tag_vector = []

	for i,row in tags.iterrows():
		tags.set_value(i,'timestamp', 1/(current_time-datetime.strptime(row['timestamp'],'%Y-%m-%d %H:%M:%S')).total_seconds())

	# Get input arguments
	model = 'tfidf'

	# Calculate total actors
	actor_ids = movie_actor.actorid.unique()
	total_actors = len(actor_ids)

	# Find movieids of the movies acted by the actor
	movieids = movie_actor.where(movie_actor['actorid']==actor_id).dropna().loc[:,'movieid']
	movietags = tags.where(tags['movieid'].isin(movieids)).dropna()
	for i,row in movietags.iterrows():
		movie_actor1 = movie_actor[(movie_actor['actorid']==actor_id)]
		movie_actor1 = movie_actor1[movie_actor1['movieid']==row[1]]
	
		row[3] /= movie_actor1.iloc[0,2]
	distinct_tags = movietags.tagid.unique()
	total_tag_weight = movietags['timestamp'].sum()

	# Find the tag weight for each distinct tag
	for tag in distinct_tags:
		movieid_with_tag = movietags.where(movietags['tagid']==tag).movieid.unique()
		actors_dfs = []
		for movie in movieid_with_tag:
			actors_dfs.append(movie_actor.where(movie_actor['movieid']==movie).dropna())
		actors_df = pd.concat(actors_dfs)
		actors = len(actors_df.actorid.unique())
	
		tag_weight = movietags.loc[movietags['tagid']==tag]['timestamp'].sum()
		if(model == 'tfidf'):
			tag_vector.append([tags_details.where(tags_details['tagId']==tag).dropna().iloc[0,1], (tag_weight/total_tag_weight)*log(total_actors/actors)])
		if(model == 'tf'):
			tag_vector.append([tags_details.where(tags_details['tagId']==tag).dropna().iloc[0,1], (tag_weight/total_tag_weight)])
		tag_vector= sorted(tag_vector, key=itemgetter(1), reverse=True)

	
	return tag_vector
