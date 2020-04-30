import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import tensorflow_hub as hub
import sys
import matplotlib.pyplot as plt
import numpy as np


import pandas as pd

from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

from players.codemaster import codemaster
import collections

class ai_codemaster(codemaster):

	def __init__(self, brown_ic=None, glove_vecs=None, word_vectors=None):
		self.cm_wordlist = []
		with open('players/cm_wordlist.txt') as infile:
			for line in infile:
				self.cm_wordlist.append(line.rstrip())
		
		print(len(self.cm_wordlist))

		self.elmo = hub.Module("https://tfhub.dev/google/elmo/3", trainable=True)
		cm_wordlist_embedding = self.elmo(self.cm_wordlist, signature="default", as_dict=True)["default"]

		with tf.compat.v1.Session() as sess:
			init = tf.compat.v1.global_variables_initializer()
			t_init = tf.compat.v1.tables_initializer()
			sess.run(init)
			sess.run(t_init)
			self.cm_wordlist_run = sess.run(cm_wordlist_embedding)

	def receive_game_state(self, words, maps):
		self.words = words
		self.maps = maps
		
	def give_clue(self):
		self.parseBoard()
		print("RED:\t", self.red_words)

		chosen_clue, chosen_num = self.getNearest(self.red_words, self.bad_words)

		print('chosen_clue is:', chosen_clue)
		return (chosen_clue, chosen_num)

	def parseBoard(self):
		self.red_words = []
		self.bad_words = []

		for i in range(25):
			if self.words[i][0] == '*':
				continue
			elif self.maps[i] == "Assassin" or self.maps[i] == "Blue" or self.maps[i] == "Civilian":
				self.bad_words.append(self.words[i].lower())
			else:
				self.red_words.append(self.words[i].lower())


	def cosineSimilarity(self, embedding, input_words, num_results):
		with tf.compat.v1.Session() as sess:
			init = tf.compat.v1.global_variables_initializer()
			t_init = tf.compat.v1.tables_initializer()
			sess.run(init)
			sess.run(t_init)

			results = []

			for i in range(len(input_words)):
				search_vect = sess.run(embedding[i]).reshape(1, -1)
				cosine_similarities = pd.Series(cosine_similarity(search_vect, self.cm_wordlist_run).flatten())

				words = []
				for index, probablity in cosine_similarities.nlargest(num_results).iteritems():
					words.append(self.cm_wordlist[index])
				
				clean_up = words[1:]
				clean_up = [w for w in clean_up if not w.startswith(input_words[i])]
				results.append(clean_up)

				# print(input[i])
				# print(words[1:])
				# print("")
			
			return results

	def getNearest(self, red_words, bad_words):
		np.set_printoptions(threshold=sys.maxsize)
		red_embeddings = self.elmo(red_words, signature="default", as_dict=True)["default"]

		results_returned = 200
		results = self.cosineSimilarity(red_embeddings, red_words, results_returned)
			
		for i in range(0, results_returned):
			common_words = collections.Counter(x for xs in results for x in xs[:i])
			# common_words = set(results[0]).intersection(*results[1:])

			if common_words:
				word, freq = common_words.most_common(1)[0]

				if freq > 1 or len(red_words) == 1:
					return word, freq