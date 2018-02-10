import pandas as pd
import os
import cPickle
import util


def main():
	mlmovies = util.read_mlmovies()
	mlratings = util.read_mlratings()
	mltags = util.read_mltags()
	movies_list = mlmovies.movieid.unique()
	ratings_list = mlratings.rating.unique()
	tags_list = mltags.tagid.unique()
	average_movie_rating = mlratings.groupby(['movieid'])['rating'].mean().reset_index()
	average_movie_rating = average_movie_rating.rename(columns={'rating':'rating_avg'})
	tag_movie_rating_grouped = mlmovies.merge(mlratings,on='movieid', how='inner').merge(mltags,on='movieid', how='inner').merge(average_movie_rating,on='movieid', how='inner')
	tag_movie_rating_tensor = []
	for tag in tags_list:
		movie_rating_matrix = []
		for movie in movies_list:
			movie_rating_list = []
			for rating in ratings_list:
				if tag_movie_rating_grouped[(tag_movie_rating_grouped.tagid == tag) & 
				(tag_movie_rating_grouped.movieid == movie) & (tag_movie_rating_grouped.rating_avg <= rating)].empty:
					movie_rating_list.append(0.0)
				else:
					movie_rating_list.append(1.0)
			movie_rating_matrix.append(movie_rating_list)
		tag_movie_rating_tensor.append(movie_rating_matrix)
	cPickle.dump( tag_movie_rating_tensor, open( "tag_movie_rating_tensor.pkl", "wb" ) )

if __name__ == "__main__":
    main()