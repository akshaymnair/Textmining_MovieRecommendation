from datetime import datetime
import pandas as pd
from numpy import log
import os

db_folder_path = os.path.join(os.path.dirname(__file__), "..", "..", "Phase2_data")
output_folder = os.path.join(os.path.dirname(__file__), "..", "..", "Output")

actor_info_file = 'imdb-actor-info.csv'


def read_actor_info():
	return pd.read_csv(os.path.abspath(os.path.join(db_folder_path, actor_info_file)))


def print_output(task, concepts):
	print('Top 3 latent semantic of ' + task + ', is :' + '\n\n')
	for idx, concept in enumerate(concepts):
		print ('\nConcept '+ str(idx+1)+' is: ')
		print('\n%50s\t %10s\t' %('Actor', 'Weight \n'))
		for row in concept:
			print ('%50s\t %10s\t' %(row[0], row[1]))

def write_output_file(task, concepts, filename):
	f = open(os.path.abspath(os.path.join(output_folder, filename)),'w')
	f.write('Top 3 latent semantic of ' + task + ', is :' + '\n')
	for idx, concept in enumerate(concepts):
		f.write('\nConcept '+ str(idx+1)+' is: ')
		f.write('\n%40s\t %15s\t' %('Actor', 'Weight \n'))
		for row in concept:
			f.write('\n%50s\t %10s\t' %(row[0], row[1]))

def append_output_file(actor_list, centroids, labels,filename):
	f = open(os.path.abspath(os.path.join(output_folder, filename)),'a')
	f.write('\n\n')
	f.write("Centroids of 3 new clusters: \n")
	for i in centroids:
		f.write(str(i))
	f.write('\n\n')
	f.write("Clustered actors into the 3 groups (0 : 1 :2) \n")
	for i, j in zip(actor_list, labels):
		f.write ('\n% 50s\t is in Cluster: %10s\t' %(i, j))

