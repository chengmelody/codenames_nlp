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
		self.num = 0
		self.elmo = hub.Module("https://tfhub.dev/google/elmo/3", trainable=True)

		self.sess = tf.compat.v1.Session()
		init = tf.compat.v1.global_variables_initializer()
		t_init = tf.compat.v1.tables_initializer()
		self.sess.run(init)
		self.sess.run(t_init)

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
		# print(new_words)
		board_embedding = self.elmo(new_words, signature="default", as_dict=True)["default"]

		board_run = self.sess.run(board_embedding)

		embedding = self.elmo(input, signature="default", as_dict=True)["default"]
		results_returned = 100

		search_vect = self.sess.run(embedding[0]).reshape(1, -1)
		cosine_similarities = pd.Series(cosine_similarity(search_vect, board_run).flatten())

		words = []
		for index, probablity in cosine_similarities.nlargest(results_returned).iteritems():
			words.append((probablity, new_words[index]))

		return words
	
	def __del__(self):
		self.sess.close()