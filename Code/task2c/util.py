import pandas as pd
import os

db_folder_path = os.path.join(os.path.dirname(__file__), "..", "..", "Phase2_data")
output_folder = os.path.join(os.path.dirname(__file__), "..", "..", "Output")

mlmovies_file = 'mlmovies.csv'
movie_actor_file = 'movie-actor.csv'
imdb_actor_info_file = 'imdb-actor-info.csv'

############### HELPER FUNCTION TO READ FILES #################################

# import mlmovies
def read_mlmovies():
	return pd.read_csv(os.path.abspath(os.path.join(db_folder_path, mlmovies_file)))

# import imdb-actor-info
def read_imdb_actor_info():
	return pd.read_csv(os.path.abspath(os.path.join(db_folder_path, imdb_actor_info_file)))

# import movie-actor
def read_movie_actor():
	return pd.read_csv(os.path.abspath(os.path.join(db_folder_path, movie_actor_file)))

################## HELPER FUNCTION TO PROCESS AND RETRIEVE ######################

def print_output(concepts, entity):
	print('The latent ' + entity + ' semantics are :')
	for idx, concept in enumerate(concepts):
		print ('\nThe latent semantic ' + str(idx+1) +' is: ')
		print('%40s\t%15s\t' %(entity, 'Weight'))
		for row in concept:
			print ('%40s\t%15s\t' %(row[0], row[1]))

def write_output_file(concepts, filename, entity):
	f = open(os.path.abspath(os.path.join(output_folder, filename)),'a')
	f.write('The latent actor semantics are :' + '\n')
	for idx, concept in enumerate(concepts):
		f.write('\nThe latent semantic ' + str(idx+1) +' is: ')
		f.write('\n%40s\t%15s\t' %(entity, 'Weight'))
		for row in concept:
			f.write('\n%40s\t%15s\t' %(row[0], row[1]))
	f.close()