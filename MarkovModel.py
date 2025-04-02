import numpy as np
import random
class MarkovModel():
    def __init__(self):
        self.states = ["R", "Pr", "Sc"]
        self.transition_matrix = np.zeros((3, 3))

    def play(self, rounds=100):
        last_move = None
        won_games = {1: 0, 0: 0, -1: 0}
        for r in range(rounds):
            opponent_move = random.choice([0,1,2])
            print(self.states[opponent_move])
            if last_move is not None:
                # [from][to] - [row][column]
                self.transition_matrix[last_move][opponent_move] += 1
                # print(self.transition_matrix)
            if last_move is not None:
                prediction = self.__predict(self.transition_matrix, last_move)
                print(f"Prediction: {prediction}")
                result = self.determine_result(prediction, self.states[last_move])
                print(f"Round result: {result}")
                won_games[result] += 1
            last_move = opponent_move
        return self.transition_matrix, last_move, won_games

    def __predict(self, transition_matrix, last_move: int) -> str:
        state_index = last_move
        row_sum = np.sum(transition_matrix[state_index])
        if row_sum > 0:
            prob = transition_matrix[state_index] / row_sum
            max_prob_index = np.argmax(prob)  # argument of maximum
            predicted_state = self.states[max_prob_index]
            max_prob = prob[max_prob_index]
            print(f'maxProb: {max_prob:.4f} for state: {predicted_state}')
            return predicted_state

    def determine_result(self, predicted_move, last_move):
        result_map = {
            ("R", "R"): 0,
            ("R", "Pr"): 1,
            ("R", "Sc"): -1,
            ("Pr", "R"): -1,
            ("Pr", "Pr"): 0,
            ("Pr", "Sc"): 1,
            ("Sc", "R"): 1,
            ("Sc", "Pr"): -1,
            ("Sc", "Sc"): 0
        }
        return result_map.get((predicted_move, last_move))

game = MarkovModel()
tm, lm, wg = game.play()
print("\nFinal Transition Matrix:")
print(game.transition_matrix)
print(f"\n{wg}")
