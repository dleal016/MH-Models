# File:         StrategyQLearner.py
# Author:       Daniel Leal
# Contact:      dannyyleall@gmail.com
# Description:  This file contains a Dyna-Q strategy algorithm in an
#               object-oriented class structure.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import QLearner
import seaborn as sns
from Utilities import *
from finta import TA

class StrategyLearner(object):
    
    BUY = 1
    HOLD = 0
    SELL = -1

    def __init__(
        self,
        accuracyBool=False, 
        numOfShares=1, 
        epochs=100, 
        numOfSteps=10,
        impact=0.0, 
        commission=0.0, 
        verbose=False,
        learner=QLearner(numOfStates=3000, numOfActions=3)):
        """
        Description: This function serves to create a StrategyLearner that can learn a trading policy.
        Parameters:
            numOfShares (int): Number of shares that can be traded in one order.
            epochs (int): The number of times to train the learner.
            numOfSteps (int): Steps in getting discretization thresholds.
            impact (float): Difference between learner and actual data.
            commision (float): Amount charged per transaction.
            verbose (bool): Print info or not.
            learner (object): Learner to implement strategy on.
        """

        self.numOfShares = numOfShares
        self.epochs = epochs
        self.numOfSteps = numOfSteps
        self.impact = impact
        self.commision = commission
        self.verbose = verbose
        self.accuracyBool = accuracyBool
        self.QLearner = learner


    def GetTechnicalIndicators(self, df):
        """
        Description: This function implements the technical indicators and features of a position and feeds it into the Q Learner.
        Parameters:
            df (pd.DataFrame): Dataframe to compute tech indicators.
        """
        df['MOMENTUM'] = TA.MOM(df, period=5)
        df['SMA'] = TA.SMA(df, period=5)
        df['BBWIDTH'] = TA.BBWIDTH(df, period=5) 
        df.drop(columns={'Open', 'High', 'Low', 'Close', 'Volume'}, inplace=True)
        df.dropna(inplace=True)
        return df


    def GetThresholds(self, dfTechnicalIndicators, numOfSteps):
        """
        Description: Computes thresholds to be used for discretization and 
        returns a 2-d numpy array where the first dimension indicates the index
        of features in dfTechnicalIndicators and second dimension refers to the value
        of a feature at the particular threshold.
        Parameters:
            dfTechnicalIndicators (pd.DataFrame): Dataframe with technical indicators applied.
            numOfSteps (int): Number of steps.
        """
        stepSize = round(dfTechnicalIndicators.shape[0] / numOfSteps)

        tempDf = dfTechnicalIndicators.copy()

        thresholds = np.zeros(shape=(dfTechnicalIndicators.shape[1], numOfSteps))

        for idx, features in enumerate(dfTechnicalIndicators.columns):
            tempDf.sort_values(by=[features], inplace=True)

            for step in range(numOfSteps):
                if step < numOfSteps - 1:
                    thresholds[idx, step] = tempDf[features].iloc[(step + 1) * stepSize]

                else:
                    thresholds[idx, step] = tempDf[features].iloc[-1]

        return thresholds


    def Discretize(self, dfTechnicalIndicators, nonNegativePosition, thresholds):
        """
        Description: This function serves to discretize the upcoming values of the deep q network. 
        In applied mathematics, discretization is the process of transferring continuous functions, 
        models, variables, and equations into discrete counterparts. This process is usually 
        carried out as a first step toward making them suitable for numerical evaluation and 
        implementation on digital computers
        Parameters:
            dfTechnicalIndicators (pd.DataFrame): Dataframe with technical indicators.
            nonNegativePosition (int): Positions of DQN.
            thresholds (float): Threshold computed from previous function.
        """
        state = nonNegativePosition * pow(self.numOfSteps, len(dfTechnicalIndicators))

        for idx in range(len(dfTechnicalIndicators)):
            threshold = thresholds[idx][thresholds[idx] >= dfTechnicalIndicators[idx]][0]

            thresholdIdx = np.where(thresholds == threshold)[1][0]

            state += thresholdIdx * pow(self.numOfSteps, idx)

        return state


    def GetPosition(self, prevPosition, signal):
        """
        Description: This function serves to find a new position based on the previous 
        position and the given signal. Signal is the action that results from querying
        a state which comes from discretize in the q table. Action is either 0, 1, 2 and is 
        the second index of the q table.
        Parameters:
            prevPosition (int): Previous Position.
            signal (int): Act.
        """
        newPosition = self.HOLD

        if prevPosition < self.BUY and signal == self.BUY:
            newPosition = self.BUY

        elif prevPosition > self.SELL and signal == self.SELL:
            newPosition = self.SELL

        return newPosition


    def GetDailyReward(self, prevPrice, currPrice, position):
        """
        Description: This function serves to calculate the daily reward of the dataframe
        as a percentage change in prices.
        Parameters:
            prevPrice (double): Previous price.
            currPrice (double): Current price.
            position (int): How many owned positions.
        """

        return position * ((currPrice / prevPrice) - 1)

    
    def CheckConverged(self, cumReturns, patience=10):
        """
        Description: This function serves to check if the cumulative returns has converged. 
        Patience is the number of epochs with no improvements in cumulative returns. This
        will return either true or false.
        Parameters:
            cumReturns (list): Cumulative returns.
            patience (int): Patience value.
        """

        if patience > len(cumReturns):
            return False

        lastFewReturns = cumReturns[-patience:]

        if len(set(lastFewReturns)) == 1:
            return True
        
        maxReturn = max(cumReturns)

        if maxReturn in lastFewReturns:
            if maxReturn not in cumReturns[:len(cumReturns) - patience]:
                return False

            else:
                return True

        return True


    def CreateDfTrades(self, orders, numOfShares, hold=0, buy=1, sell=-1):
        """
        Description: This function serves to simply create a dataframe for 
        orders executed to simulate trading.
        Parameters:
            orders (pd.DataFrame): Orders dataframe.
            numOfShares (int): Number of shares.
            hold (int): Value for holding.
            buy (int): Value for buying.
            sell (int): Value for selling.
        """

        trades = []

        if self.accuracyBool == False:
            buyOrSell = orders[orders != hold]

            for date in buyOrSell.index:
                if buyOrSell.loc[date] == buy:
                    trades.append((date, numOfShares))

                elif buyOrSell.loc[date] == sell:
                    trades.append((date, -numOfShares))

        elif self.accuracyBool == True:
            buyOrSell = orders
            
            for date in buyOrSell.index:
                if buyOrSell.loc[date] == buy:
                    trades.append((date, numOfShares))

                elif buyOrSell.loc[date] == sell:
                    trades.append((date, -numOfShares))

                elif buyOrSell.loc[date] == hold:
                    trades.append((date, 0))

        dfTrades = pd.DataFrame(trades, columns=["Date", "Shares"])
        dfTrades.set_index("Date", inplace=True)

        return dfTrades

    def CreateBmDfTrades(self, symbol, startDate, endDate, numOfShares):
        """
        Description: This function serves to simply createa an empty df that can be used to test
        against the dfTrades dataframe and establishes a benchmark.
        Parameters:
            symbol (string): Stock acronym.
            startDate (string): Start date.
            endDate (string): End date.
            numOfShares (int): Number of shares.
        """

        bmPrices = NormalizeDfs(IndividualHistoricalData(symbol, startDate, endDate, keepAllColumns="NO"))[0]

        dfBmTrades = pd.DataFrame(
            data=[
                (bmPrices.index.min(), numOfShares),
                (bmPrices.index.max(), -numOfShares)
            ], 
            columns=["Date", "Shares"]
        )

        dfBmTrades.set_index("Date", inplace=True)

        return dfBmTrades

    def SymbolValueFromTrading(
        self, 
        dfOrders, 
        symbol, 
        startDate, 
        endDate,
        startVal=1, 
        commision=9.95, 
        impact=0.05):
        """
        Description: This function serves to simulate trading a stock 
        based on the orders performed and symbol given. This returns
        a column of the portfolio value given that one stock after every 
        action performed.
        Parameters:
            dfOrders (pd.DataFrame): Trade orders dataframe.
            symbol (string): Stock acronym.
            startDate (string): Start date.
            endDate (string): End date.
            startVal (int): Starting value.
            commission (double): Trading commission.
            impact (double): Trading impact.
        """
        dfOrders.sort_index(ascending=True, inplace=True)

        dfPrices = NormalizeDfs(IndividualHistoricalData(symbol, startDate, endDate, keepAllColumns="NO"))[0]

        dfPrices["Cash"] = 1.0

        dfPrices.fillna(method="ffill", inplace=True)
        dfPrices.fillna(method="bfill", inplace=True)
        dfPrices.fillna(1.0, inplace=True)

        dfTrades = pd.DataFrame(np.zeros((dfPrices.shape)), dfPrices.index,
                                dfPrices.columns)

        for idx, row in dfOrders.iterrows():
            tradedShareVal = dfPrices.loc[idx, symbol] * row["Shares"]
            transactionCost = commision + impact * dfPrices.loc[idx, symbol] * abs(row["Shares"])

            if row["Shares"] > 0:
                dfTrades.loc[idx, symbol] = dfTrades.loc[idx, symbol] + row["Shares"]
                dfTrades.loc[idx, "Cash"] = dfTrades.loc[idx, "Cash"] - tradedShareVal - transactionCost

            elif row["Shares"] < 0:
                dfTrades.loc[idx, symbol] = dfTrades.loc[idx, symbol] + row["Shares"]
                dfTrades.loc[idx, "Cash"] = dfTrades.loc[idx, "Cash"] - tradedShareVal - transactionCost

        dfHoldings = pd.DataFrame(np.zeros((dfPrices.shape)), dfPrices.index,
                                  dfPrices.columns)

        for rowCount in range(len(dfHoldings)):
            if rowCount == 0:
                dfHoldings.iloc[0, :-1] = dfTrades.iloc[0, :-1].copy()
                dfHoldings.iloc[0, -1] = dfTrades.iloc[0, -1] + startVal

            else:
                dfHoldings.iloc[rowCount] = dfHoldings.iloc[rowCount - 1] + dfTrades.iloc[rowCount]

            rowCount += 1

        dfVal = dfPrices * dfHoldings
        
        portVals = pd.DataFrame(dfVal.sum(axis=1), dfVal.index, ["Port Val"])
        
        return portVals

    def MarketSimulator(
        self, 
        dfOrders, 
        dfOrdersBm, 
        symbol,
        startDate, 
        endDate, 
        title, 
        startVal=1,
        commission=9.95, 
        impact=0.005,
        saveFigure=False, 
        figName="Plot.png",
        showPlot=False):
        """
        Description: This function intakes dfOrders dataframe that executes the trades 
        and displays the portfolio value respectively.
        Parameters:
            dfOrders (pd.DataFrame): Trade orders dataframe.
            dfOrdersBm (pd.DataFrame): Trade orders benchmark dataframe.
            symbol (string): Stock acronym.
            startDate (string): Start date.
            endDate (string): End date.
            commission (double): Trading commission.
            impact (double): Trading impact.
            saveFigure (bool): Save figure boolean.
            figName (string): Name for saved figure.
            showPlot (bool): Show plot boolean.
        """
        portVals = self.SymbolValueFromTrading(dfOrders=dfOrders, symbol=symbol,
                                               startVal=startVal, startDate=startDate,
                                               endDate=endDate, commision=commission,
                                               impact=impact)

        portValsBm = self.SymbolValueFromTrading(dfOrders=dfOrdersBm, symbol=symbol,
                                                 startVal=startVal, startDate=startDate,
                                                 endDate=endDate, commision=commission,
                                                 impact=impact)

        portValsBm.rename(columns={"Port Val": symbol}, inplace=True)

        temp = []
        temp.append(portVals)
        temp.append(portValsBm)
        df = temp[0].join(temp[1])

        plt.plot(df.loc[:, df.columns[1]], label=symbol, color="maroon")
        plt.plot(df.loc[:, df.columns[0]], label="QLearner", color="darkgreen")
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("Normalized Prices")
        plt.legend(loc="best")

        fig = plt.gcf()
        fig.set_size_inches(9, 4)
        
        if saveFigure:
            plt.savefig(figName)

        if showPlot:
            plt.show()

        plt.close()
        
    def AddEvidence(
        self,
        symbol='GLD',
        startDate="2021-01-01", 
        endDate="2022-01-01",
        startVal=1):
        """
        Description: This function serves to add training data to the 
        Strategy learner.
        Parameters:
            symbol (string): Stock acronym.
            startDate (string): Start date.
            endDate (string): End date.
            startVal (int): Starting value.
        """
        tempDf = NormalizeDfs(IndividualHistoricalData(symbol=symbol, startDate=startDate,
                                      endDate=endDate, keepAllColumns="YES"))[0]

        dfPrices = NormalizeDfs(IndividualHistoricalData(symbol=symbol, startDate=startDate,
                                      endDate=endDate, keepAllColumns="NO"))[0]

        dfFeatures = self.GetTechnicalIndicators(tempDf)

        dfThres = self.GetThresholds(dfFeatures, self.numOfSteps)

        cumReturns = []

        for epoch in range(1, self.epochs + 1):

            # Initial position is hold.
            position = self.HOLD

            # Create pandas series that captures order signals.
            orders = pd.Series(index=dfFeatures.index)

            for day, date in enumerate(dfFeatures.index):
                # Get a state.
                state = self.Discretize(dfFeatures.loc[date],
                                        position + 1,
                                        dfThres)

                # Get action, do not update table if first time.
                if date == dfFeatures.index[0]:
                    action = self.QLearner.Act(state, 0.0, update=False)

                # Otherwise, calculate reward and update table.
                else:
                    prevPrice = dfPrices[symbol].iloc[day - 1]
                    currPrice = dfPrices[symbol].loc[date]

                    reward = self.GetDailyReward(prevPrice, currPrice, position)
                    action = self.QLearner.Act(state, reward, 
                                               done=date==dfFeatures.index[-1],
                                               update=True)

                # If last day, sell.
                if date == dfFeatures.index[-1]:
                    newPosition = -position

                else:
                    newPosition = self.GetPosition(position, action - 1)

                orders.loc[date] = newPosition

                position += newPosition

            dfTrades = self.CreateDfTrades(orders, self.numOfShares)
            portVals = self.SymbolValueFromTrading(dfOrders=dfTrades,
                                                   symbol=symbol,
                                                   startDate=startDate,
                                                   endDate=endDate,
                                                   startVal=startVal,
                                                   commision=self.commision,
                                                   impact=self.impact)

            cr = portVals.iloc[-1, 0] / portVals.iloc[0, 0] - 1
            cumReturns.append(cr)

            if self.verbose:
                print("Epoch: {}, Cumulative Return: {}\n".format(epoch, cr))

            if epoch > 20:
                if self.CheckConverged(cumReturns):
                    break

        if self.verbose:
            sns.heatmap(self.QLearner.Q, cmap='Blues')
            plt.plot(cumReturns)
            plt.xlabel("Epochs")
            plt.ylabel("Cumulative Return (%)")
            plt.show()
    
    def Query(
        self,
        symbol='GLD',
        startDate="2022-01-02",
        endDate=datetime.today().strftime('%Y-%m-%d')):
        """
        Description: This function serves to test the existing policy on a new data set.
        Parameters:
            symbol (string): Stock acronym.
            startDate (string): Start date.
            endDate (string): End date.
        """

        tempDf = NormalizeDfs(IndividualHistoricalData(symbol=symbol,
                                            startDate=startDate,
                                            endDate=endDate,
                                            keepAllColumns="YES"))[0]                

        dfFeatures = self.GetTechnicalIndicators(tempDf)

        thresholds = self.GetThresholds(dfTechnicalIndicators=dfFeatures,
                                        numOfSteps=self.numOfSteps)

        position = self.HOLD

        orders = pd.Series(index=dfFeatures.index)

        for date in dfFeatures.index:
            state = self.Discretize(dfFeatures.loc[date],
                                    position + 1,
                                    thresholds)

            action = self.QLearner.Act(state, 0.0, update=False)

            if date == dfFeatures.index[-1]:
                newPosition = -position

            else:
                newPosition = self.GetPosition(position, action - 1)

            orders.loc[date] = newPosition

            position += newPosition

        dfTrades = self.CreateDfTrades(orders=orders,
                                       numOfShares=self.numOfShares)

        return dfTrades
        