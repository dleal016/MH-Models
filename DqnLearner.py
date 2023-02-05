# File:         DqnLearner.py
# Author:       Daniel Leal
# Contact:      dannyyleall@gmail.com
# Description:  This file contains a Dyna-Q deep learning algorithm in an
#               object-oriented class structure with neural networks.

import numpy as np
import pandas as pd
import random as rand
import matplotlib.pyplot as plt
from collections import deque
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

class DeepQNetwork(object):
    
    ACTIONS = { 0:-1, 1:0, 2:1 }

    def __init__(
        self,
        stateSize=20, 
        actionSize=20, 
        alpha=0.001, 
        gamma=0.95, 
        epsilon=1.0,
        minEpsilon=0.9,
        epsilonDecay=0.9,
        verbose=False):
        """
        Description: Constructor for deep nueral network q learner.
        Params:
            stateSize(int): Number of states.
            actionSize (int): Number of actions.
            alpha (float): Learning rate.
            gamma (float): Value of future reward.
            epsilon (float): Exploration rate.
            minEpsilon (float): Minimum exploration rate.
            epsilonDecay (float): Decay rate for exploration.
            verbose (bool): Print info out.
        """

        self.stateSize = stateSize
        self.actionSize = actionSize
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.minEpsilon = minEpsilon
        self.epsilonDecay = epsilonDecay
        self.memory = deque(maxlen=2000)
        
        # Build neural network.
        self.model = self.BuildModel()
        
        if verbose:
            self.verbose = 1
        
        else:
            self.verbose = 0

    def BuildModel(self):
        """
        Description: Builds the neural networks for our deep-q learning model.
        """
        # Prepares/Initializes NN layers.
        model = Sequential()
        model.add(Dense(60, input_dim=self.stateSize, activation='relu'))
        model.add(Dense(60, activation='relu'))
        model.add(Dense(3, activation='linear'))
        
        # Configures model for training using MSE loss function and Adam optimizer.
        model.compile(loss='mse', optimizer=Adam(lr=self.alpha))

        return model

    def Remember(self, state, action, reward, nextState, done):
        """
        Description: Appends information to memory to remember performance and values.
        Params:
            state (int): Current state.
            action (int): Current action.
            reward (int): Current reward.
            nextState (int): Next state.
            done (bool): Completed or not.
        """
        self.memory.append((state, action, reward, nextState, done))

    def Act(self, state):
        """
        Description: Makes decision on best action to perform based on state parameter.
        Params:
            state (int): Current state.
        """
        if np.random.rand() <= self.epsilon:
            return rand.randrange(self.actionSize)

        actVals = self.model.predict(np.asarray([state]), verbose=self.verbose)

        return np.argmax(actVals[0])

    def RewardTarget(self, memory):
        """
        Description: Makes model understand what the best outcome is and reward algorithm accordingly.
        Params:
            memory (deque): Model memory.
        """

        states = []
        targRwds = []

        for state, action, reward, nextState, done in memory:
            target = reward

            if not done:
                target = (
                    reward + self.gamma *
                    np.amax(self.model.predict(np.asarray([nextState]), 
                    verbose=self.verbose)[0])
                    )
            
            tempTarget = self.model.predict(np.asarray([state]), verbose=self.verbose)
            tempTarget[0][action] = target

            states.append(state)
            targRwds.append(tempTarget)

        return (states, targRwds)
    
    def Query(self, df):
        """
        Description: Fits model.
        Params:
            df (pd.DataFrame): Data to be tested.
        """
        validMem = self.AddEvidence(df, df.index[-1], memory=[])

        validStates, validTargetRewards = self.RewardTarget(validMem)

        dfStates, dfTargetRewards = self.RewardTarget(self.memory)

        self.model.fit(np.asarray(dfStates),
                       np.asarray(dfTargetRewards)[:, 0, :],
                       validation_data=(np.asarray(validStates), np.asarray(validTargetRewards)[:, 0, :]),
                       epochs=10, 
                       verbose=self.verbose)

        if self.epsilon > self.minEpsilon:
            self.epsilon *= self.epsilonDecay

    def AddEvidence(self, df, endDate, memory=[]):
        """
        Description: Creates memory for training purposes.
        Param:
            df (pd.DataFrame): Dataframe to create memory from.
            endDate (string): End data of dataframe.
            memory (deque): Memory.
        """
        

        for idx, (domain, range) in enumerate(df.iloc[self.stateSize:-2].iterrows()):

            state = np.asarray(df.iloc[idx: idx + self.stateSize]['Return'])

            action = self.ACTIONS[self.Act(state)]

            nextState = np.asarray(df.iloc[idx + 1: idx + self.stateSize + 1]['Return'])

            done = domain == endDate

            reward = action * df.iloc[idx + self.stateSize]['Return']

            memory.append((state, action, reward, nextState, done))
        
        return memory

    def TransformDf(self, df, windowSize=2):
        """
        Description: Transforms dataframe into a tradeable one using daily returns for the 
        DQN to use as reward system.
        Params:
            df (pd.DataFrame): Dataframe to transform.
            windowSize (int): Window size for return computing.
        """
        
        df['Return'] = df[df.columns[0]].rolling(
            window=windowSize).apply(lambda x: x[1] / x[0] - 1)

        return df

    def CreateTradesDf(self, df, learner):
        """
        Description: Creates trades dataframe.
        Params:
            df (pd.DataFrame): Dataframe to transform.
            learner (DqnLearner): Learner instance.
        """

        dfTrades = { "Trade": []}
        cumRet = 1

        df = df.append(df.iloc[-1])

        for idx, (domain, range) in enumerate(
            df[learner.stateSize:-2].iterrows()):
        
            state = np.asarray(df.iloc[
                idx: idx + learner.stateSize
            ]['Return'])
            
            position = learner.Act(state)
            
            reward = position * df.iloc[ 
                idx + learner.stateSize + 1
            ]['Return']

            dfTrades["Trade"].append(position)
            cumRet *= 1 + reward

        dfTrades = pd.DataFrame(dfTrades, 
                    index=df.index[
                        learner.stateSize + 1:-1]).join(df)
        dfTrades["Portfolio Return"] = (dfTrades["Trade"] * dfTrades["Return"])
        dfTrades["DQNLearner"] = (1 + dfTrades["Portfolio Return"]).cumprod()
        dfTrades[df.columns[0]] = dfTrades[df.columns[0]] / dfTrades.iloc[0][df.columns[0]]
   
        return dfTrades

    def PlotDqnPerformance(self, testTrades, symbol):
        """
        Description: Plots the performance of the DQN.
        Paramteres:
            testTrades (pd.DataFrame): The trades dataframe.
            symbol (string): The symbol acronym.
        """
        plt.plot(testTrades[[symbol]], label=symbol, color="maroon")
        plt.plot(testTrades[["DQNLearner"]], label="DQN Learner", color="green")
        plt.title("DQN Test Plot")
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.legend(loc="best")

        fig = plt.gcf()
        fig.set_size_inches(9, 4)
        sym = str(symbol).lower().capitalize()
        plt.savefig(f"Images/{sym}DqnLearnerVisual.png")
        plt.close()
