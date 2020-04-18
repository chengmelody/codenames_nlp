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

from players.guesser import guesser
import collections

class ai_guesser(guesser):

	def __init__(self, brown_ic=None, glove_vecs=None, word_vectors=None):
		self.brown_ic = brown_ic
		self.glove_vecs = glove_vecs
		self.word_vectors = word_vectors
		self.num = 0

		self.elmo = hub.Module("https://tfhub.dev/google/elmo/3", trainable=True)

	def get_board(self, words):
		self.words = words
		
	def get_clue(self, clue, num):
		self.clue = clue
		self.num = num
		self.start = num
		print("The clue is:", clue, num, sep=" ")
		return [clue, num]

	def keep_guessing(self, clue, board):
		return self.num > 0

	def give_answer(self):
		result = self.getNearest([self.clue])
		print(result)
		self.num -= 1
		return result[0][1]
	
	def getNearest(self, input):
		np.set_printoptions(threshold=sys.maxsize)

		new_words = [x for x in self.words if "*" not in x]
		board_embedding = self.elmo(new_words, signature="default", as_dict=True)["default"]

		with tf.compat.v1.Session() as sess:
			init = tf.compat.v1.global_variables_initializer()
			t_init = tf.compat.v1.tables_initializer()
			sess.run(init)
			sess.run(t_init)
			self.board_run = sess.run(board_embedding)


		embeddings = self.elmo(input, signature="default", as_dict=True)["default"]

		results_returned = 5

		with tf.compat.v1.Session() as sess:
			init = tf.compat.v1.global_variables_initializer()
			t_init = tf.compat.v1.tables_initializer()
			sess.run(init)
			sess.run(t_init)

			search_vect = sess.run(embeddings[0]).reshape(1, -1)

			cosine_similarities = pd.Series(cosine_similarity(search_vect, self.board_run).flatten())
			words = []
			for index, prob in cosine_similarities.nlargest(results_returned).iteritems():
				words.append((prob, new_words[index]))
			
		return words
			# # clean_up = words[1:]
			# # clean_up = [w for w in clean_up if not w.startswith(input[i])]
			# results.append(words)

			# # 	print(input[i])
			# # 	print(words[1:])
			# # 	print("")
			
			# # print(results)

			# 	common_words = collections.Counter(x for xs in results for x in xs[:i])
			# 	# common_words = set(results[0]).intersection(*results[1:])

			# 	if common_words:
			# 		word, freq = common_words.most_common(1)[0]

			# 		if freq > 1 or len(input) == 1:
			# 			return word, freq

