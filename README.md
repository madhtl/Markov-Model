MarkovModel1 is slightly changed version from MarkovModel which I initially implemented, offering 
a more versatile view on the problem.

Hidden Markov model offers a way to find patterns in unobserved modes of the opponent in the Gameplay,
which on bigger scale may resolve eg energetic distribution dilemmas, if such parameters were identifiable.
For now the implementation of the "tactic detection" rules are heuristic.
The logic is based on 3 matrices ([state][state], [hidden_state][state],[hidden_state][hidden_state]), probability of occuring values and normalization.
Stochastic model, thus having some randomity variable- we are only calculating the probabiltiy of an on-going trend, 
it's not an axiom.

Suppose last 5 observed moves were: [R, R, R, Pr, Sc]. 
The pattern for first 3 moves = "Repetitive", then it switches to "Countering".
Here useful would be to add Viterbi's logic for decoding the strategy, where the states can change frequently.

For repetitive strategy the diagonal will have argmax with lowest prob 50%-if changing every 2nd round 
the move eg [P P R R Sc Sc]

If the opponent beats at least 4 times in the row the previous move, it's apparent he uses the 
Countering technique, used here the determine_result f from Markov_Model file.

