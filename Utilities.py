# File:         Utilities.py
# Author:       Daniel Leal
# Contact:      dannyyleall@gmail.com
# Description:  This file contains all helper functions.

import yfinance as yf
from datetime import datetime 

def NormalizeDfs(dfArray):
    """"
    Description: Normalizes the array of dataframes.
    Parameters:
        dfArray (array): Array of dataframes.
    """
    normalizedDfs = []
    for df in range(len(dfArray)):
        tempDf = dfArray[df]
        normal = tempDf/tempDf.iloc[0,:]
        normalizedDfs.append(normal)
    
    return normalizedDfs

def IndividualHistoricalData(symbol, startDate, endDate=datetime.today().strftime('%Y-%m-%d'), keepAllColumns="Yes"):
    """"
    Description: Retrieves historical data..
    Parameters:
            symbol (string): Stock acronym.
            startDate (string): Start date.
            endDate (string): End date.
            keepAllColumns (string): Keeps all columns from API.
    """
    # Returns dataframe for data of symbol.
    data = yf.Ticker(symbol)
    dataDf = data.history(period='1d', start=startDate, end=endDate)

    if keepAllColumns.lower() == "yes":
        dataDf = dataDf.drop(columns={'Dividends', 'Stock Splits'})
        return dataDf
    
    elif keepAllColumns.lower() == 'no':
        # Only keep row with 'close' prices.
        dataDf = dataDf.iloc[: , [3]].copy()
        dataDf = dataDf.rename(columns={'Close':symbol})
        return dataDf

def HistoricalData(symbolsArray, startDate, endDate=datetime.today().strftime('%Y-%m-%d'), keepAllColumns="Yes"):

    """"
    Description: Retrieves historical data..
    Parameters:
            symbol (string): Stock acronym.
            startDate (string): Start date.
            endDate (string): End date.
            keepAllColumns (string): Keeps all columns from API.
    """

    dfs = []

    # Returns dfArray that iterated through symbols to get data for each.
    for symbol in range(len(symbolsArray)):
        tempDf = IndividualHistoricalData(symbolsArray[symbol], startDate, endDate, keepAllColumns)
        dfs.append(tempDf)

    return dfs
