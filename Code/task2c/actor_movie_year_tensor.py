import pandas as pd
import os
import cPickle
import util


def main():
	mlmovies = util.read_mlmovies()
	imdb_actor_info = util.read_imdb_actor_info()
	movie_actor = util.read_movie_actor()
	movies_list = mlmovies.movieid.unique()
	year_list = mlmovies.year.unique()
	actor_list = imdb_actor_info.id.unique()
	movie_year_matrix = []

	actor_movie_year_grouped = pd.merge(movie_actor, mlmovies, on=['movieid','movieid'], how='inner')
	actor_movie_year_tensor = []
	count=0
	for actor in actor_list:
		movie_year_matrix = []
		for movie in movies_list:
			movie_year_list = []
			for year in year_list:
				if actor_movie_year_grouped[(actor_movie_year_grouped.actorid == actor) & 
				(actor_movie_year_grouped.movieid == movie) & (actor_movie_year_grouped.year == year)].empty:
					movie_year_list.append(0.0)
				else:
					movie_year_list.append(1.0)
			movie_year_matrix.append(movie_year_list)
		actor_movie_year_tensor.append(movie_year_matrix)
	cPickle.dump( actor_movie_year_tensor, open( "actor_movie_year_tensor.pkl", "wb" ) )

if __name__ == "__main__":
    main()