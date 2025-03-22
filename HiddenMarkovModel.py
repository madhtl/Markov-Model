import numpy as np
import random
class HiddenMarkovModel():
    def __init__(self):
        self.states = ["R", "Pr", "Sc"]
        self.transition_matrix = np.zeros((3, 3))

    def play(self, rounds=100):
        last_move = None
        for r in range(rounds):
            opponent_move = random.choice([0,1,2])
            print(self.states[opponent_move])
            if last_move is not None:
                # [from][to] - [row][column]
                self.transition_matrix[last_move][opponent_move] += 1
                # print(self.transition_matrix)
            last_move = opponent_move
        return self.transition_matrix, last_move

    def predict(self, transition_matrix, last_move : int) -> str:
        state_index = last_move
        row_sum = np.sum(transition_matrix[state_index])
        if row_sum > 0:
            prob = transition_matrix[state_index] / row_sum
            max_prob_index = np.argmax(prob)  # argument of maximum
            predicted_state = self.states[max_prob_index]
            max_prob = prob[max_prob_index]
            print(f'maxProb: {max_prob:.4f} for state: {predicted_state}')
            return predicted_state

game = HiddenMarkovModel()
tm, lm = game.play()
print("\nFinal Transition Matrix:")
print(game.transition_matrix)
predicted_move = game.predict(tm, lm)
print(f"Predicted next move is {predicted_move}")
