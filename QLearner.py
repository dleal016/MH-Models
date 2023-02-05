# File:         QLearner.py
# Author:       Daniel Leal
# Contact:      dannyyleall@gmail.com
# Description:  This file contains a Dyna-Q machine learning algorithm in an
#               object-oriented class structure.

from copy import deepcopy
from collections import deque
import numpy as np
import random as rand

class QLearner(object):

    def __init__(
        self,
        numOfStates=100,
        numOfActions=4,
        alpha=0.2,
        gamma=0.9, 
        rar=0.5,
        radr=0.99,
        dyna=0,
        verbose=False):
        """
        Description: This function serves as the constructor for a Dyna QLearner instance.
        Parameters:
            numOfStates (int): Number of states within a Q Table.
            numOfActions (int): Number of actions within a Q Table.
            alpha (float): Value for learning rate.
            gamma (float): Value of future reward.
            rar (float): Random action rate (Probability of selection a random action at each step).
            radr (float): Random action decay rate (After each update, rar = rar * radr).
            dyna (int): Number of dyna updates.
            verbose (bool): Display info or not.
        """
        self.numOfStates = numOfStates
        self.numOfActions = numOfActions
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna
        self.verbose = verbose

        # Double ended queue data structure that allows insert and delete at both ends.
        self.memory = deque(maxlen=2000)

        # Keep track of the latest state and action.
        self.state = 0
        self.action = 0

        # Initialize a Q-table that records and updates q values for each action/state.
        self.Q = np.zeros(shape=(numOfStates, numOfActions))

        # Keep track of transitions from s to sprime when performing an aciton in Dyna-Q.
        self.T = {}

        # Keep track of reward for each action in each state when doing Dyna-Q.
        self.R = np.zeros(shape=(numOfStates, numOfActions))

    def RememberQValues(self, state, action, reward, nextState, done):
        """
        Description: Remembers the Q values and appends to deque data structure.
        Parameters:
            state (int): State of Q table.
            action (int): Action to perform for respective state.
            reward (float): Reward for specific aciton.
            nextState (int): Subsequent state of Q table.
            done (bool): If q value acquisition is complete.
        """
        self.memory.append((state, action, reward, nextState, done))
        
    def Act(self, state, reward, done=False, update=True):
        """
        Description: Peforms a query operation depending on current status of Q-table.
        Parameters:
            state (int): Current state to perform query on.
            reward (float): Immediate reward from previous action.
            done (bool): If acting has been performed.
            update (bool): Update Q table based on values.
        """
        if update:
            return self.Query(state, reward, done=done)
        
        else:
            return self.QueryState(state)

    def QueryState(self, state):
        """
        Description: Find the next action to take in state s. Update the latest state and action 
        without updating the Q table.
        Parameters:
            state (int): The new state
        """
        if rand.uniform(0.0, 1.0) < self.rar:
            action = rand.randint(0, self.numOfActions - 1)
        
        else:
            action = self.Q[state, :].argmax()

        self.state = state
        self.action = action

        if self.verbose:
            print("\nState = {}, Action = {}".format(state, action))

        return action

    def Query(self, statePrime, reward, done=False):
        """
        Description: Find the next action to take in state s-prime. Update the latest state 
        and action and the Q table. Update rule:
        Q'[s, a] = (1 - α) · Q[s, a] + α · (r + γ · Q[s', argmax a'(Q[s', a'])]).
        Parameters:
            statePrime (int): New state.
            reward (float): Immediate reward for taking the previous action.
        """
        self.RememberQValues(self.state, self.action, reward, statePrime, done)

        # Update Q table.
        self.Q[self.state, self.action] = (
            (1 - self.alpha) * self.Q[self.state, self.action] + 
            self.alpha * (reward + self.gamma * self.Q[statePrime, self.Q[statePrime, :].argmax()])
        )

        # Implement Dyna-Q.
        if self.dyna > 0:
            # Update reward table.
            self.R[self.state, self.action] = (
                (1 - self.alpha) * self.R[self.state, self.action] + self.alpha * reward
            )

            if (self.state, self.action) in self.T:
                if statePrime in self.T[(self.state, self.action)]:
                    self.T[(self.state, self.action)][statePrime] += 1

                else:
                    self.T[(self.state, self.action)][statePrime] = 1

            else:
                self.T[(self.state, self.action)] = { statePrime: 1 }

            Q = deepcopy(self.Q)

            # Hallucinations.
            for i in range (self.dyna):
                dummyState = rand.randint(0, self.numOfStates - 1)
                dummyAction = rand.randint(0, self.numOfActions - 1)

                if (dummyState, dummyAction) in self.T:
                    # Find the most common statePrime as a result of taking action.
                    dummyStatePrime = max(self.T[(dummyState, dummyAction)], key=lambda x: self.T[(dummyState, dummyAction)][x])

                    # Update temp table.
                    Q[dummyState, dummyAction] = (
                        (1 - self.alpha) * Q[dummyState, dummyAction] + 
                        self.alpha * (self.R[dummyState, dummyAction] + 
                        self.gamma * Q[dummyStatePrime, Q[dummyStatePrime, :].argmax()])
                    )

            # Update once dyna is complete.
            self.Q = deepcopy(Q)

        # Find the next action to take and update.
        nextAction = self.QueryState(statePrime)
        self.rar *= self.radr

        if self.verbose:
            print("\nState = {}, Action = {}, Reward = {}".format(statePrime, nextAction, reward))

        return nextAction
