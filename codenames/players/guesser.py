class guesser():
    words = 0
    clue = 0
    clues = []
    
    def __init__(self, brown_ic=None, glove_vecs=None, word_vectors=None, elmo_embedding=None):
        pass


class human_guesser(guesser):

    def __init__(self):
        pass

    def get_clue(self, clue, num):
        print("The clue is:", clue, num, sep=" ")

    def get_board(self, words):
        self.words = words

    def give_answer(self):
        answer_input = input("Guesser makes turn.\nPlease enter a valid Word >> ")
        type(answer_input)

        while not self.is_valid(answer_input):
            print("Input Invalid")
            print(self.words)
            answer_input = input("Please enter a valid Word >> ")
            type(answer_input)
        return answer_input

    def keep_guessing(self, clue, num):
        return True

    def is_valid(self, result):
        if result.upper() in self.words:
            return True
        else:
            return False
