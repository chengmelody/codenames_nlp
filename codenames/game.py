from players.codemaster import *
from players.guesser import *
from nltk.stem import WordNetLemmatizer
from nltk.stem.lancaster import LancasterStemmer
from nltk.corpus import wordnet
from nltk.corpus import words
from nltk.corpus import wordnet_ic
import numpy as np
import scipy
import itertools
import importlib
import random
import array
import os
import sys
import colorama
import gensim.models.keyedvectors as word2vec
import gensim.downloader as api
import argparse

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import tensorflow_hub as hub


class Game:
	guesser = 0
	codemaster = 0

	def __init__(self, codemaster, guesser, glove_vecs=None, elmo=None):
		codemaster_module = importlib.import_module(codemaster)
		guesser_module = importlib.import_module(guesser)

		self.codemaster = codemaster_module.ai_codemaster(glove_vecs=glove_vecs, elmo_embedding=elmo)
		self.guesser = guesser_module.ai_guesser(glove_vecs=glove_vecs, elmo_embedding=elmo)

		self.seed = 'time'

		f = open("game_wordpool.txt", "r")
		
		if f.mode == 'r':
			temp_array = f.read().splitlines()
			self.words = set([])
			# if duplicates were detected and the set length is not 25 then restart
			while len(self.words) != 25:
				self.words = set([])
				for x in range(0, 25):
					random.shuffle(temp_array)
					self.words.add(temp_array.pop())
			self.words = list(sorted(self.words))
			random.shuffle(self.words)

		self.maps = ["Red"]*8 + ["Blue"]*7 + ["Civilian"]*9 + ["Assassin"]
		random.shuffle(self.maps)
	
	# def __init__(self):
	# 	parser = argparse.ArgumentParser(
  #       description="Run the Codenames AI competition game.",
  #       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	# 	parser.add_argument("codemaster", help="Path to codemaster package or 'human'")
	# 	parser.add_argument("guesser", help="Path to guesser package or 'human'")
	# 	parser.add_argument("--seed", help="Random seed value for board state -- integer or 'time'",default='time')
	# 	parser.add_argument("--w2v", help="Path to w2v file or 'none'",default='none')
	# 	parser.add_argument("--glove", help="Path to glove file or 'none'",default='none')
	# 	parser.add_argument("--wordnet", help="Name of wordnet file or 'none', most like ic-brown.dat",default='none')
		
	# 	args = parser.parse_args()

	# 	# if the game is going to have an ai, load up word vectors
	# 	if sys.argv[1] != "human" or sys.argv[2] != "human":
	# 		brown_ic = None
	# 		if args.wordnet != 'none':
	# 			brown_ic = wordnet_ic.ic(args.wordnet)
	# 		glove_vecs = {}
	# 		if args.glove != 'none':
	# 			with open(args.glove, encoding="utf-8") as infile:
	# 				for line in infile:
	# 					line = line.rstrip().split(' ')
	# 					glove_vecs[line[0]] = np.array([float(n) for n in line[1:]])
	# 			print('loaded glove vectors')
	# 		word_vectors = {}
	# 		if args.w2v != 'none':
	# 			word_vectors = word2vec.KeyedVectors.load_word2vec_format(
	# 				args.w2v, binary=True, unicode_errors='ignore')
	# 			print('loaded word vectors')

	# 	if args.codemaster == "human":
	# 		self.codemaster = human_codemaster()
	# 		print('human codemaster')
	# 	else:
	# 		codemaster_module = importlib.import_module(args.codemaster)
	# 		self.codemaster = codemaster_module.ai_codemaster(brown_ic, glove_vecs, word_vectors)
	# 		print('loaded codemaster')

	# 	if args.guesser == "human":
	# 		self.guesser = human_guesser()
	# 		print('human guesser')
	# 	else:
	# 		guesser_module = importlib.import_module(args.guesser)
	# 		self.guesser = guesser_module.ai_guesser(brown_ic, glove_vecs, word_vectors)
	# 		print('loaded guesser')

	# 	self.seed = 'time'
	# 	if args.seed != 'time':
	# 		self.seed = args.seed
	# 		random.seed(int(args.seed))

	# 	f = open("game_wordpool.txt", "r")
		
	# 	if f.mode == 'r':
	# 		temp_array = f.read().splitlines()
	# 		self.words = set([])
	# 		# if duplicates were detected and the set length is not 25 then restart
	# 		while len(self.words) != 25:
	# 			self.words = set([])
	# 			for x in range(0, 25):
	# 				random.shuffle(temp_array)
	# 				self.words.add(temp_array.pop())
	# 		self.words = list(sorted(self.words))
	# 		random.shuffle(self.words)

	# 	self.maps = ["Red"]*8 + ["Blue"]*7 + ["Civilian"]*9 + ["Assassin"]
	# 	random.shuffle(self.maps)

	# prints out board with color-paired words, only for codemaster, color && stylistic
	def display_board_codemaster(self):
		print(str.center("___________________________BOARD___________________________\n", 60))
		counter = 0
		for i in range(len(self.words)):
			if counter >= 1 and i % 5 == 0:
				print("\n")
			if self.maps[i] is 'Red':
				print(str.center(colorama.Fore.RED + self.words[i], 15), " ", end='')
				counter += 1
			elif self.maps[i] is 'Blue':
				print(str.center(colorama.Fore.RESET + self.words[i], 15), " ", end='')
				counter += 1
			elif self.maps[i] is 'Civilian':
				print(str.center(colorama.Fore.RESET + self.words[i], 15), " ", end='')
				counter += 1
			else:
				print(str.center(colorama.Fore.MAGENTA + self.words[i], 15), " ", end='')
				counter += 1
		print(str.center(colorama.Fore.RESET + 
			"\n___________________________________________________________", 60))
		print("\n")

	# prints the list of words in a board like fashion (5x5)
	def display_board(self):
		print(colorama.Style.RESET_ALL)
		print(str.center("___________________________BOARD___________________________", 60))
		counter = 0
		for i in range(len(self.words)):
			if i % 5 == 0:
				print("\n")
			print(str.center(self.words[i], 10), " ", end='')

		print(str.center("\n___________________________________________________________", 60))
		print("\n")

	# aesthetic purposes, doesn't impact function of code.
	def display_map(self):
		print("\n")
		print(str.center(colorama.Fore.RESET + 
			"____________________________MAP____________________________\n", 55))
		counter = 0
		for i in range(len(self.maps)):
			if counter >= 1 and i % 5 == 0:
				print("\n")
			if self.maps[i] is 'Red':
				print(str.center(colorama.Fore.RED + self.maps[i], 15), " ", end='')
				counter += 1
			elif self.maps[i] is 'Blue':
				print(str.center(colorama.Fore.RESET + self.maps[i], 15), " ", end='')
				counter += 1
			elif self.maps[i] is 'Civilian':
				print(str.center(colorama.Fore.RESET + self.maps[i], 15), " ", end='')
				counter += 1
			else:
				print(str.center(colorama.Fore.MAGENTA + self.maps[i], 15), " ", end='')
				counter += 1
		print(str.center(colorama.Fore.RESET + 
			"\n___________________________________________________________", 55))
		print("\n")

	def list_words(self):
		return self.words

	def list_map(self):
		return self.maps

	# takes in an int index called guess to compare with the Map
	def accept_guess(self,guess_index):
		# CodeMaster will always win with Red and lose if Blue =/= 7 or Assassin == 1
		if self.maps[guess_index] == "Red":
			self.words[guess_index] = "*Red*"
			if self.words.count("*Red*") >= 8:
				return "Win"
			return "Hit_Red"
		elif self.maps[guess_index] == "Blue":
			self.words[guess_index] = "*Blue*"
			if self.words.count("*Blue*") >= 7:
				return "Lose"
			else:
				return "Still Going"
		elif self.maps[guess_index] == "Assassin":
			self.words[guess_index] = "*Assassin*"
			return "Lose"
		else:
			self.words[guess_index] = "*Civilian*"
			return "Still Going"

	def cls(self):
		print('\n'*4)

	def write_results(self, num_of_turns, win):
		red_result = 0
		blue_result = 0
		civ_result = 0
		assa_result = 0

		for i in range(len(self.words)):
			if self.words[i] == "*Red*":
				red_result += 1
			elif self.words[i] == "*Blue*":
				blue_result += 1
			elif self.words[i] == "*Civilian*":
				civ_result += 1
			elif self.words[i] == "*Assassin*":
				assa_result += 1
		total = red_result + blue_result + civ_result + assa_result

			# f = open("bot_results.txt", "a")
			# # if successfully opened start appending
			# if f.mode == 'a':
			# 	f.write(
			# 		f'TOTAL:{num_of_turns} B:{blue_result} C:{civ_result} A:{assa_result}'
			# 		f' R:{red_result} CM:{sys.argv[1]} GUESSER:{sys.argv[2]} SEED:{self.seed}\n'
			# 		)
			# f.close()
		
		return Result(win, num_of_turns, blue_result, civ_result, assa_result, red_result)

	def run(self):
		game_condition = "Hit_Red"
		game_counter = 0
		while game_condition != "Lose" or game_condition != "Win":
			# board setup and display
			#self.cls()
			words_in_play = self.list_words()
			current_map = self.list_map()
			self.codemaster.receive_game_state(words_in_play, current_map)
			# self.display_map()
			#self.display_board_codemaster()
			# codemaster gives clue & number here
			clue, num = self.codemaster.give_clue()
			game_counter += 1
			keep_guessing = True
			guess_num = 0
			num = int(num)

			#self.cls()
			self.guesser.get_clue(clue, num)
			
			game_condition = "Hit_Red"
			while guess_num <= num and keep_guessing and game_condition == "Hit_Red":
				self.guesser.get_board(words_in_play)
				guess_answer = self.guesser.give_answer()

				# if no comparisons were made/found than retry input from codemaster
				if guess_answer == "no comparisons":
					break
				guess_answer_index = words_in_play.index(guess_answer.upper().strip())
				game_condition = self.accept_guess(guess_answer_index)

				if game_condition == "Hit_Red":
					#self.cls()
					#self.display_board_codemaster()
					guess_num += 1
					#print("Keep Guessing?")
					keep_guessing = self.guesser.keep_guessing(clue, words_in_play)
					# if(keep_guessing):
						#print("The clue is :", clue, num, sep=" ")

				# if guesser selected a civilian or a blue-paired word
				elif game_condition == "Still Going":
					break

				elif game_condition == "Lose":
					#self.display_board_codemaster()
					#print("You Lost")
					game_counter = 25
					# self.write_results(game_counter)
					#print("Game Counter:", game_counter)
					return self.write_results(game_counter, False)

				elif game_condition == "Win":
					#self.display_board_codemaster()
					#print("You Won")
					# self.write_results(game_counter)
					#print("Game Counter:", game_counter)
					return self.write_results(game_counter, True)

class Result:
	def __init__(self, win, num_of_turns, blue_result, civ_result, assa_result, red_result):
		self.win = win
		self.num_of_turns = num_of_turns
		self.blue_result = blue_result
		self.civ_result = civ_result
		self.assa_result = assa_result
		self.red_result = red_result

class Stats:
	def __init__(self, runs, cm, g):
		self.runs = 0
		self.total_runs = runs
		self.cm = cm
		self.g = g

		self.wins = 0
		self.loses = 0

		self.total_num_of_turns = 0
		self.total_blue_result = 0
		self.total_civ_result = 0
		self.total_assa_result = 0
		self.total_red_result = 0

		self.min_num_of_turns = float('inf')
		self.min_blue_result = float('inf')
		self.min_civ_result = float('inf')
		self.min_assa_result = float('inf')
		self.min_red_result = float('inf')

		self.max_num_of_turns = float('-inf')
		self.max_blue_result = float('-inf')
		self.max_civ_result = float('-inf')
		self.max_assa_result = float('-inf')
		self.max_red_result = float('-inf')

		self.avg_num_of_turns = 0
		self.avg_blue_result = 0
		self.avg_civ_result = 0
		self.avg_assa_result = 0
		self.avg_red_result = 0
	
	def addStat(self, r):
		self.runs += 1

		if r.win:
			self.wins += 1
		else:
			self.loses += 1

		self.total_num_of_turns += r.num_of_turns
		self.total_blue_result += r.blue_result
		self.total_civ_result += r.civ_result
		self.total_assa_result += r.assa_result
		self.total_red_result += r.red_result

		self.min_num_of_turns = min(self.min_num_of_turns, r.num_of_turns)
		self.min_blue_result = min(self.min_blue_result, r.blue_result)
		self.min_civ_result = min(self.min_civ_result, r.civ_result)
		self.min_assa_result = min(self.min_assa_result, r.assa_result)
		self.min_red_result = min(self.min_red_result, r.red_result)

		self.max_num_of_turns = max(self.max_num_of_turns, r.num_of_turns)
		self.max_blue_result = max(self.max_num_of_turns, r.blue_result)
		self.max_civ_result = max(self.max_num_of_turns, r.civ_result)
		self.max_assa_result = max(self.max_num_of_turns, r.assa_result)
		self.max_red_result = max(self.max_num_of_turns, r.red_result)
	
	def computeAverage(self):
		self.avg_num_of_turns = self.total_num_of_turns / self.runs
		self.avg_blue_result = self.total_blue_result / self.runs
		self.avg_civ_result = self.total_civ_result / self.runs
		self.avg_assa_result = self.total_assa_result / self.runs
		self.avg_red_result = self.total_red_result / self.runs

	def printStats(self, write):
		self.computeAverage()
		s = 	(f'Run:{self.runs} '
					f'Wins:{self.wins} Loses:{self.loses} '
					f'Min Turns:{self.min_num_of_turns} Avg Turns:{self.avg_num_of_turns} Max Turns:{self.max_num_of_turns} '
					f'Min Blue:{self.min_blue_result} Avg Blue:{self.avg_blue_result} Max Blue:{self.max_blue_result} '
					f'Min Red:{self.min_red_result} Avg Red:{self.avg_red_result} Max Red:{self.max_red_result} '
					f'Min Civ:{self.min_civ_result} Avg Civ:{self.avg_civ_result} Max Civ:{self.max_civ_result} '
					f'Min Assa:{self.min_assa_result} Avg Assa:{self.avg_assa_result} Max Assa:{self.max_assa_result} '
					f'CM:{self.cm} GUESSER:{self.g}\n')

		# print(s)

		forfile = 	(f'{self.runs}, '
			f'{self.wins}, {self.loses}, '
			f'{self.min_num_of_turns}, {self.avg_num_of_turns}, {self.max_num_of_turns}, '
			f'{self.min_blue_result}, {self.avg_blue_result}, {self.max_blue_result}, '
			f'{self.min_red_result}, {self.avg_red_result}, {self.max_red_result}, '
			f'{self.min_civ_result}, {self.avg_civ_result}, {self.max_civ_result}, '
			f'{self.min_assa_result}, {self.avg_assa_result}, {self.max_assa_result}, '
			f'{self.cm}, {self.g}\n')

		if write:
			f = open("bot_results.txt", "a")
			# if successfully opened start appending
			if f.mode == 'a':
				f.write(forfile)
			f.close()

def createElmoEmbedding():
	cm_wordlist = []
	with open('players/cm_wordlist.txt') as infile:
		for line in infile:
			cm_wordlist.append(line.rstrip())
	
	elmo = hub.Module("https://tfhub.dev/google/elmo/3", trainable=True)
	cm_wordlist_embedding = elmo(cm_wordlist, signature="default", as_dict=True)["default"]

	with tf.compat.v1.Session() as sess:
		init = tf.compat.v1.global_variables_initializer()
		t_init = tf.compat.v1.tables_initializer()
		sess.run(init)
		sess.run(t_init)
		return sess.run(cm_wordlist_embedding)

if __name__ == "__main__":
	codemasters = ["bert"]
	guessers = ["glove"]
	total_runs = 30

	glove_vecs = {}
	with open("players/glove.6B.100d.txt", encoding="utf-8") as infile:
		for line in infile:
			line = line.rstrip().split(' ')
			glove_vecs[line[0]] = np.array([float(n) for n in line[1:]])
	print('loaded glove vectors')

	# elmo_embedding = createElmoEmbedding()

	for cm in codemasters:
		for g in guessers:
			runs = 0
			s = Stats(total_runs, cm, g)
			while runs <= total_runs:
				runs += 1
				game = Game("players.codemaster_"+ cm, "players.guesser_" + g, glove_vecs)
				result = game.run()
				s.addStat(result)
				s.printStats(True)

	# for g in guessers:
	# 	runs = 0
	# 	s = Stats(total_runs, "elmo", g)
	# 	while runs <= total_runs:
	# 		runs += 1
	# 		game = Game("players.codemaster_elmo", "players.guesser_" + g, glove_vecs, elmo_embedding)
	# 		result = game.run()
	# 		s.addStat(result)
	# 		s.printStats(True)
	# 	s.printStats(True)