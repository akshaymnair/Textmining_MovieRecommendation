import pandas as pd
import os

db_folder_path = os.path.join(os.path.dirname(__file__), "..", "..", "Phase2_data")
output_folder = os.path.join(os.path.dirname(__file__), "..", "..", "Output")

mlmovies_file = 'mlmovies.csv'
mlratings_file = 'mlratings.csv'
mltags_file = 'mltags.csv'
genome_tags_file = 'genome-tags.csv'

############### HELPER FUNCTION TO READ FILES #################################

# import mlmovies
def read_mlmovies():
	return pd.read_csv(os.path.abspath(os.path.join(db_folder_path, mlmovies_file)))

# import mlratings
def read_mlratings():
	return pd.read_csv(os.path.abspath(os.path.join(db_folder_path, mlratings_file)))

# import mltags
def read_mltags():
	return pd.read_csv(os.path.abspath(os.path.join(db_folder_path, mltags_file)))

# import mltags
def read_genome_tags():
	return pd.read_csv(os.path.abspath(os.path.join(db_folder_path, genome_tags_file)))

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
	f.write('The latent '+ entity + ' semantics are :' + '\n')
	for idx, concept in enumerate(concepts):
		f.write('\nThe latent semantic ' + str(idx+1) +' is: ')
		f.write('\n%40s\t%15s\t' %(entity, 'Weight'))
		for row in concept:
			f.write('\n%40s\t%15s\t' %(row[0], row[1]))
	f.close()

def print_partition(partitions):
	for patition_index, partition in partitions.iteritems():
		print('\nThe partition ' + str(patition_index) + ' contains')
		for entity, entity_values in partition.iteritems():
			print('\n%s\t' %(entity))
			for value in entity_values:
				print('%40s\t' %(value))

def write_partition_output_file(partitions, filename):
	f = open(os.path.abspath(os.path.join(output_folder, filename)),'a')
	for patition_index, partition in partitions.iteritems():
		f.write('\nThe partition ' + str(patition_index) + ' contains')
		for entity, entity_values in partition.iteritems():
			f.write('\n%s\t' %(entity))
			for value in entity_values:
				f.write('%40s\t' %(value))