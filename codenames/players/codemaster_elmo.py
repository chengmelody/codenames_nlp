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
		self.previous_clue = ''
		with open('players/cm_wordlist.txt') as infile:
			for line in infile:
				self.cm_wordlist.append(line.rstrip())

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
		red_words, bad_words, lose_words = self.parseBoard()
		print("RED:\t", red_words)

		chosen_clue, chosen_num = self.getNearest(red_words, bad_words, lose_words)
		self.previous_clue = chosen_clue

		print('chosen_clue is:', chosen_clue)
		return (chosen_clue, chosen_num)

	def parseBoard(self):
		red_words = []
		bad_words = []
		lose_words = []

		# Categorize the board into three categories
		for i in range(25):
			if self.words[i][0] == '*':
				continue
			elif self.maps[i] == "Blue" or self.maps[i] == "Civilian":
				bad_words.append(self.words[i].lower())
			elif self.maps[i] == "Assassin":
				lose_words.append(self.words[i].lower())
			else:
				red_words.append(self.words[i].lower())

		return red_words, bad_words, lose_words

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
			
			return results

	def removeBadClues(self, red_common_words, bad_common_words, lose_closest):
		diff_common_words = red_common_words - bad_common_words

		for w in lose_closest[0]:
			del diff_common_words[w]
			
		if self.previous_clue in diff_common_words:
			diff_common_words[self.previous_clue] -= 1

		for clue in self.previous_clue:
		 	del diff_common_words[clue]

		# print(diff_common_words.most_common(5))
		return diff_common_words

	def getNearest(self, red_words, bad_words, lose_words):
		np.set_printoptions(threshold=sys.maxsize)
		red_embeddings = self.elmo(red_words, signature="default", as_dict=True)["default"]
		bad_embeddings = self.elmo(bad_words, signature="default", as_dict=True)["default"]
		lose_embeddings = self.elmo(lose_words, signature="default", as_dict=True)["default"]

		results_returned = 200
		red_results = self.cosineSimilarity(red_embeddings, red_words, results_returned)
		bad_results = self.cosineSimilarity(bad_embeddings, bad_words, results_returned)
		lose_results = self.cosineSimilarity(lose_embeddings, lose_words, results_returned)

		new_common_words = None
		for i in range(30, results_returned):
			red_common_words = collections.Counter(x for xs in red_results for x in xs[:i])
			bad_common_words = collections.Counter(x for xs in bad_results for x in xs[:i])

			new_common_words = self.removeBadClues(red_common_words, bad_common_words, lose_results)
			if new_common_words:
				word, freq = new_common_words.most_common(1)[0]

				if freq > 1 or len(red_words) == 1:
					return word, freq
		else:
			new_common_words.most_common(1)[0]
			return word, freq