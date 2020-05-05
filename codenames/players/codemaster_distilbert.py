import collections
from sentence_transformers import SentenceTransformer
import scipy

from players.codemaster import codemaster

class ai_codemaster(codemaster):

	def __init__(self, brown_ic=None, glove_vecs=None, word_vectors=None, elmo_embedding=None):
		self.cm_wordlist = []
		self.previous_clue = ''
		with open('players/cm_wordlist.txt') as infile:
			for line in infile:
				self.cm_wordlist.append(line.rstrip())

		self.model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
		self.cm_wordlist_embedding = self.model.encode(self.cm_wordlist)

	def receive_game_state(self, words, maps):
		self.words = words
		self.maps = maps
		
	def give_clue(self):
		red_words, bad_words, lose_words = self.parseBoard()
		print("RED:\t", red_words)

		chosen_clue, chosen_num = self.getBestClue(red_words, bad_words, lose_words)
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


	def getClosestWords(self, input, num_results):
		input_embedding = self.model.encode(input)

		answer = []
		for query, query_embedding in zip(input, input_embedding):
				distances = scipy.spatial.distance.cdist([query_embedding], self.cm_wordlist_embedding, "cosine")[0]

				results = zip(range(len(distances)), distances)
				results = sorted(results, key=lambda x: x[1])
				# print(query)
				top_words = []
				for idx, distance in results[0:num_results]:
						if not self.cm_wordlist[idx].startswith(query):
							top_words.append(self.cm_wordlist[idx])
				
				# print(top_words[1:])
				answer.append(top_words[1:])
				
		return answer

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


	def getBestClue(self, red_words, bad_words, lose_words):
		num_results = 200
		red_closest = self.getClosestWords(red_words, num_results)
		bad_closest = self.getClosestWords(bad_words, num_results)
		lose_closest = self.getClosestWords(lose_words, num_results)

		red_common_words = None
		for i in range(30, num_results):
			red_common_words = collections.Counter(x for xs in red_closest for x in xs[:i])
			bad_common_words = collections.Counter(x for xs in bad_closest for x in xs[:i])

			new_common_words = self.removeBadClues(red_common_words, bad_common_words, lose_closest)
			if red_common_words:
				word, freq = red_common_words.most_common(1)[0]

				if freq > 1 or len(red_words) == 1:
					return word, freq
		else:
			red_common_words.most_common(1)[0]
			return word, freq
