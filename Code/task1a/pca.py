import pandas as pd
from sklearn.decomposition import PCA
import sys
import util

def main():
	#Read and display the input, 
	genre = sys.argv[1]
	no_of_components = 4
	genome_tags = util.read_genome_tags()
	#print genome_tags

	tf_idf_matrix = util.get_tf_idf_matrix(genre)
	tagid_list = list(tf_idf_matrix.columns.values)
	tag_list = genome_tags[genome_tags['tagId'].isin(tagid_list)]['tag'].tolist()
	#print tag_list
	
	pca = PCA(n_components=no_of_components)
	pca.fit(tf_idf_matrix)

	concepts = []
	for i in range(no_of_components):
		concept = []
		for j, component in enumerate(pca.components_[i]):
			concept.append((tag_list[j], component))
		print ('\nThe ' + str(i+1) +'th concept is: ')
		util.print_output(concept)
		concepts.append(concept)



	#print('For genre: ' + genre + ', output is :')
if __name__ == "__main__":
    main()
