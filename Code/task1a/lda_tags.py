#CSE515 project phase 2
#author Akshay
import sys
import pandas as pd
import warnings
import os
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim
from gensim import corpora, models

db_folder_path = os.path.join(os.path.dirname(__file__), "..", "..", "Phase2_data")
output_folder = os.path.join(os.path.dirname(__file__), "..", "..", "Output")

mltags_file = 'mltags.csv'
mlmovies_file = 'mlmovies.csv'
genome_tags_file = 'genome-tags.csv'

output_file = 'task1a_lda_tags.out.txt'

mltags = pd.read_csv(os.path.abspath(os.path.join(db_folder_path, mltags_file)))
movie_genre = pd.read_csv(os.path.abspath(os.path.join(db_folder_path, mlmovies_file)))
tags_details = pd.read_csv(os.path.abspath(os.path.join(db_folder_path, genome_tags_file)))
tag_vector = []
#print(mltags)

movie_genre = pd.DataFrame(movie_genre.genres.str.split('|').tolist(), index = movie_genre.movieid).stack()
movie_genre = movie_genre.reset_index()[[0,'movieid']]
movie_genre.columns = ['genres','movieid']

genre = sys.argv[1]
#genre = 'Drama'
movieids = (movie_genre[movie_genre['genres']==genre])
genre_tagid = pd.merge(movieids,mltags, on='movieid')
genre_tags = pd.merge(genre_tagid, tags_details, left_on=['tagid'], right_on=['tagId'])

documents = []
dist_movies = genre_tags.movieid.unique()
for i in dist_movies:
    documents.append((genre_tags[genre_tags['movieid']==i].dropna().loc[:,'tag']))
#list(documents)

# turn documents into a id <-> term dictionary
dictionary = corpora.Dictionary(documents)
#print(dictionary.token2id)

# convert tokenized documents into a document-term matrix 
corpus = [dictionary.doc2bow(doc) for doc in documents]
#for i in corpus:
 #   print(i)
#(termid, term frequency)

# generate LDA model
#num_topics: required. An LDA model requires the user to determine how many topics should be generated
#id2word: required. The LdaModel class requires our previous dictionary to map ids to strings
#passes: optional. The number of laps the model will take through corpus. 
#The greater the number of passes, the more accurate the model will be. A lot of passes can be slow on a very large corpus.

concepts = gensim.models.ldamodel.LdaModel(corpus, num_topics=4, id2word = dictionary, passes=200)
#print(concepts.print_topics(num_topics=4, num_words=4))
#print(concepts.print_topics())
print('For genre: ' + genre + ', Latent topics are:\n')
i=0
while i<4:
	print('Topic: '+str(i)+' is')
	print(concepts.print_topic(i))
	print('\n')
	i=i+1

f = open(os.path.abspath(os.path.join(output_folder, output_file)),'w')
f.write('For genre: ' + genre + ', Latent topics are:\n')
i=0
while i<4:
	f.write('Topic: '+str(i)+' is\n')
	f.write(concepts.print_topic(i))
	f.write('\n')
	i=i+1

#END

