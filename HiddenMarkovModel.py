from MarkovModel import determine_result
from MarkovModel1 import MarkovModel1
import numpy as np
from collections import deque
class HiddenMarkovModel:
    def __init__(self):
        self.transition_matrix = None  # hidden_state -> hidden_state
        self.emission_matrix = None  # hidden_state -> observable_state
        self.last_tactic = None
        self.states = ["R", "Pr", "Sc"]
        self.hidden_states = ["Random", "Repetitive", "Countering", "Undefined"]
        self.state_prob = {state: 1.0/len(self.hidden_states) for state in self.hidden_states}
        self.markov = MarkovModel1(states = self.states)
        self.emission_matrix = np.zeros((len(self.hidden_states), len(self.states)))
        self.moves = deque(maxlen=10)
        self.em = deque(maxlen=10)
        self.tactic_h = deque(maxlen=10)

    def start(self, state_from: str, state_to: str):
        current_tactic = "Undefined"
        self.markov.learn_transition(state_from, state_to)
        self.moves.append(state_to)
        if len(self.moves)>=10:
            tm = self.markov.transition_matrix
            self.normalize(tm)
            self.predict_observation(state_to)


    def predict_observation(self, state_from: str) -> None:
        for state in self.states:
            idx = self.states.index(state)
            row = self.markov.transition_matrix[idx]
            row_sum = np.sum(self.markov.transition_matrix[idx])
            if row_sum == 0:
                continue
        streak = 0
        max_streak = 0
        for i in range(1, len(self.moves)):
            former = self.moves[i-1]
            latter = self.moves[i]
            if determine_result(latter, former)==1:
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                streak = 0
        last_three = list(self.moves)[-3:]
        if len(last_three)!=3:
            print("The length of last_three is not 3")
        if all(m == last_three[0] for m in last_three):
            current_tactic = "Repetitive"
        elif max_streak > 4: # Any 4 consecutive "counterings" by accident are on the base of 1.23% prob
            current_tactic = "Countering"
        else:
            current_tactic = "Random"

        self.change_emission_matrix(current_tactic, state_from)
        print(f"The current tactic is: {current_tactic}")
        if self.last_tactic is not None:
            self.change_transition_matrix(self.last_tactic, current_tactic)
        self.last_tactic = current_tactic
        if len(self.tactic_h)>=20:
            pre_1 = self.predict_new_state_emission(current_tactic, self.emission_matrix)
            pre_2 = self.predict_new_tactic_transition(current_tactic, self.transition_matrix)
            print(f"The new state will be {pre_1}")
            print(f"The next hidden tactic is said to be {pre_2}")


    def predict_new_state_emission(self, current_tactic : str, matrix : np.ndarray) -> str:
        i = self.hidden_states.index(current_tactic)
        print(f"For current tactic {current_tactic}")
        row_sum = np.sum(matrix[i])
        if row_sum > 0:
            prob = matrix[i] / row_sum
            print(f"with probability {max(prob)}")
            max_prob_index = np.argmax(prob)  # argument of maximum
            next_state = self.states[max_prob_index]
            print(f"The next state is {next_state}")
        else:
            raise Exception(f"No transitions recorded from state: {current_tactic}")
        return next_state

    def predict_new_tactic_transition(self, current_tactic : str, matrix : np.ndarray) -> str:
        i = self.hidden_states.index(current_tactic)
        print(f"For current tactic {current_tactic}")
        row_sum = np.sum(matrix[i])
        if row_sum > 0:
            prob = matrix[i] / row_sum
            print(f"with probability {max(prob)}")
            max_prob_index = np.argmax(prob)  # argument of maximum
            next_state = self.hidden_states[max_prob_index]
            print(f"The next tactic is {next_state}")
        else:
            raise Exception(f"No transitions recorded from state: {current_tactic}")
        return next_state

    def change_emission_matrix(self, current_tactic : str, current_state : str)-> None:
        e_m = self.emission_matrix
        hs_idx = self.hidden_states.index(current_tactic)
        st_idx = self.states.index(current_state)
        e_m[hs_idx][st_idx]+=1
        self.em.append(current_tactic)
        if len(self.em)>=10:
            row = e_m[hs_idx]
            row_sum = np.sum(row)
            if row_sum > 0:
                # normalization
                self.normalize(e_m)
                j = np.argmax(row)
                predicted_state = self.states[j]
                prob_next_move = row / row_sum
                confidence = max(prob_next_move)
                print(f"The predicted next state is: {predicted_state} with confidence: {confidence:.2f}")
            else:
                raise Exception(f"Cannot normalize in {current_tactic}, {current_state}")

    def change_transition_matrix(self, prev_tactic: str, current_tactic : str)-> None:
        if self.transition_matrix is None:
            size = len(self.hidden_states)
            self.transition_matrix = np.zeros([size, size])
        i = self.hidden_states.index(prev_tactic)
        j = self.hidden_states.index(current_tactic)
        self.transition_matrix[i][j] += 1
        self.tactic_h.append(prev_tactic)
        if len(self.tactic_h)>=10:
            self.normalize(self.transition_matrix)

    def normalize(self, tm : np.ndarray) -> np.ndarray:
        for i in range(tm.shape[0]):
            row_sum = np.sum(tm[i])
            if row_sum > 0:
                tm[i] = tm[i] / row_sum
        return tm


HMM = HiddenMarkovModel()
if __name__ == "__main__":
    HMM.start("Pr", "Pr")
    HMM.start("Pr", "Pr")
    HMM.start("Pr", "Pr")
    HMM.start("R", "Pr")
    HMM.start("Sc", "Pr")
    HMM.start("Sc", "Sc")
    HMM.start("Pr", "R")
    HMM.start("R", "R")
    HMM.start("Pr", "Pr")
    HMM.start("Pr", "Pr")
    HMM.start("Pr", "Pr")
    HMM.start("Pr", "Pr")
    HMM.start("Sc", "Pr")
    HMM.start("Pr", "Pr")
    HMM.start("R", "Pr")
    HMM.start("Sc", "R")
    HMM.start("Pr", "Sc")
    HMM.start("R", "Pr")
    HMM.start("Sc", "R")
    HMM.start("Pr", "Sc")
    HMM.start("Pr", "Pr")
    HMM.start("Pr", "Pr")
    HMM.start("Pr", "Pr")
    HMM.start("Pr", "Pr")
    HMM.start("Pr", "Pr")
    HMM.start("Pr", "Pr")
    HMM.start("Pr", "Pr")
    HMM.start("Pr", "Pr")
    HMM.start("Pr", "Pr")
    HMM.start("Pr", "Pr")
    HMM.start("Pr", "Pr")
    HMM.start("Pr", "Pr")
    HMM.start("Pr", "Pr")
    HMM.start("R", "Pr")
    HMM.start("Sc", "Pr")
    HMM.start("Sc", "Sc")
    HMM.start("Pr", "R")
    HMM.start("R", "R")
    HMM.start("Pr", "Pr")
    HMM.start("Pr", "Pr")
    HMM.start("Pr", "Pr")
    HMM.start("Pr", "Pr")
    HMM.start("Sc", "Pr")
    HMM.start("Pr", "Pr")
    HMM.start("R", "Pr")
    HMM.start("Sc", "R")
    HMM.start("Pr", "Sc")
    HMM.start("R", "Pr")
    HMM.start("Sc", "R")
    HMM.start("Pr", "Sc")
    HMM.start("Pr", "Pr")
    HMM.start("Pr", "Pr")
    HMM.start("Pr", "Pr")
    HMM.start("Pr", "Pr")
    HMM.start("Pr", "Pr")
    HMM.start("Pr", "Pr")
    HMM.start("Pr", "Pr")
    HMM.start("Pr", "Pr")
    HMM.start("Pr", "Pr")
    HMM.start("Pr", "Pr")
    HMM.start("Pr", "Pr")
    HMM.start("Pr", "Pr")
    HMM.start("Pr", "Pr")
    HMM.start("R", "Pr")
    HMM.start("Sc", "Pr")
    HMM.start("Sc", "Sc")
    HMM.start("Pr", "R")
    HMM.start("R", "R")
    HMM.start("Pr", "Pr")
    HMM.start("Pr", "Pr")
    HMM.start("Pr", "Pr")
    HMM.start("Pr", "Pr")
    HMM.start("Sc", "Pr")
    HMM.start("Pr", "Pr")
    HMM.start("R", "Pr")
    HMM.start("Sc", "R")
    HMM.start("Pr", "Sc")
    HMM.start("R", "Pr")
    HMM.start("Sc", "R")
    HMM.start("Pr", "Sc")
    HMM.start("Pr", "Pr")
    HMM.start("Pr", "Pr")
    HMM.start("Pr", "Pr")
    HMM.start("Pr", "Pr")
    HMM.start("Pr", "Pr")
    HMM.start("Pr", "Pr")
    HMM.start("Pr", "Pr")
    HMM.start("Pr", "Pr")
    HMM.start("Pr", "Pr")
    HMM.start("Pr", "Pr")




