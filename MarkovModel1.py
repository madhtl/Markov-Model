import numpy as np
class MarkovModel1:
    def __init__(self, states = None) -> None:
        self.states = []
        self.state_index = {}
        self.transition_matrix = np.zeros((0,0))
        if states is not None:
            for state in states:
                self.add_state(state)

    def add_state(self, state):
        if state not in self.states:
            self.states.append(state)
            self.state_index[state] = len(self.states) - 1
            size = len(self.states)
            new_matrix = np.zeros((size, size))
            new_matrix[:self.transition_matrix.shape[0], :self.transition_matrix.shape[1]] = self.transition_matrix
            self.transition_matrix = new_matrix


    def predict_new_state(self, current_state : str) -> str:
            i = self.state_index[current_state]
            row_sum = np.sum(self.transition_matrix[i])
            if row_sum > 0:
                prob = self.transition_matrix[i] / row_sum
                print(f"with probability {max(prob)}")
                max_prob_index = np.argmax(prob)  # argument of maximum
                next_state = self.states[max_prob_index]
            else:
                raise Exception(f"No transitions recorded from state: {current_state}")
            return next_state

    def learn_transition(self, state_from: str, state_to: str) -> None:
        for state in [state_from, state_to]:
            self.add_state(state)
        i = self.state_index[state_from]
        j = self.state_index[state_to]
        self.transition_matrix[i][j] += 1

if __name__ == "__main__":
    mm = MarkovModel1()
    mm.learn_transition("R", "Pr")
    mm.learn_transition("Pr", "Sc")
    mm.learn_transition("Sc", "R")
    mm.learn_transition("R", "Pr")
    mm.learn_transition("R", "Sc")

    print("States:", mm.states)
    print("Transition matrix:\n", mm.transition_matrix)
    print("Predicted next from R:", mm.predict_new_state("R"))
    print("Predicted next from Pr:", mm.predict_new_state("Pr"))
