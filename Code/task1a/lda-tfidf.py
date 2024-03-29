import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation as LDA
import sys
import util
import os

output_file = 'task1a_lda.out.txt'

def main():
	if len(sys.argv) < 2:
		print('Expected arguments are not provided.')
		return
	genre = sys.argv[1]
	no_of_components = 4
	genome_tags = util.read_genome_tags()
	#print genome_tags

	tf_idf_matrix = util.get_tf_idf_matrix(genre)
	tagid_list = list(tf_idf_matrix.columns.values)
	tag_list = genome_tags[genome_tags['tagId'].isin(tagid_list)]['tag'].tolist()
	#print tag_list
	
	lda = LDA(n_components=no_of_components)
	lda.fit(tf_idf_matrix)

	concepts = []
	for i in range(no_of_components):
		concept = []
		for j, component in enumerate(lda.components_[i]):
			concept.append((tag_list[j], component))
		concept.sort(key=lambda tup: abs(tup[1]), reverse=True)
		concepts.append(concept)
	util.print_output(genre, concepts)
	util.write_output_file(genre, concepts, output_file)

if __name__ == "__main__":
    main()
