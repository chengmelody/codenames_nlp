import collections
from sentence_transformers import SentenceTransformer
import scipy

from players.guesser import guesser

class ai_guesser(guesser):

	def __init__(self, brown_ic=None, glove_vecs=None, word_vectors=None):
		self.brown_ic = brown_ic
		self.glove_vecs = glove_vecs
		self.word_vectors = word_vectors
		self.num = 0

		self.model = SentenceTransformer('roberta-base-nli-stsb-mean-tokens')

	def get_board(self, words):
		self.words = words
		
	def get_clue(self, clue, num):
		self.clue = clue
		self.num = num
		print("The clue is:", clue, num, sep=" ")
		return [clue, num]

	def keep_guessing(self, clue, board):
		return self.num > 0

	def give_answer(self):
		result = self.getBestGuess([self.clue])
		print(result)
		self.num -= 1
		return result[0][1]
	
	def getBestGuess(self, guess):
		updated_board = [x for x in self.words if "*" not in x]
		board_embedding = self.model.encode(updated_board)
		guess_embedding = self.model.encode(guess)

		results_returned = 5
		
		best_guess = []
		distances = scipy.spatial.distance.cdist([guess_embedding[0]], board_embedding, "cosine")[0]

		results = zip(range(len(distances)), distances)
		results = sorted(results, key=lambda x: x[1])

		for idx, distance in results[0:results_returned]:
				#print(updated_board[idx], "(Cosine Score: %.4f)" % (1-distance))
				best_guess.append((1-distance, updated_board[idx]))
		
		return best_guess