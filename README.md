# Multiple Assets Portfolio Optimization Risk Management (Monte-Carlo)

## Overview
This project focuses on optimizing a diversified portfolio and assessing its risk using the Monte Carlo simulation method. It is designed for financial advisors and portfolio managers who aim to balance risk and return according to clients' financial goals and risk tolerance.

## Objectives
**Portfolio Optimization:** Identify the optimal asset allocation that maximizes returns while minimizing risk.
**Risk Measurement:** Calculate the Value at Risk (VaR) using Monte Carlo simulations to understand potential losses.

## Key Concepts
**Sharpe Ratio:** A metric to evaluate portfolio efficiency by measuring the return per unit of risk.
**Value at Risk (VaR):** A statistical measure to quantify the potential loss in a portfolio, given a certain confidence level over a specified period.

## Project Structure

**1. Importing Necessary Libraries and Data Collection:**

- Download historical data for selected tickers (UKX, SPY, GLD, QQQ, BND, MSCI, VTI) using yfinance.
- Time range: Last 10 years.
```python
#importing libraries
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
import matplotlib.pyplot as plt

#Defining tickers and time range
tickers = ['UKX', 'SPY', 'GLD', 'QQQ', 'BND', 'MSCI', 'VTI']
#Setting up the end date to Today
end_date = datetime.today()
#Setting up the starting date 10 years ago
start_date = end_date - timedelta(days = 10*365)

#Downloding adjusted closing prices in a data frame
adj_close_df = pd.DataFrame()
for ticker in tickers:
    data = yf.download(ticker, start = start_date,end = end_date)
    adj_close_df[ticker] = data['Adj Close']
print(adj_close_df)
```
**2. Portfolio Optimization:**
- Calculate log returns and covariance matrix of the assets.
- Optimize the portfolio to maximize the Sharpe ratio using scipy.optimize.minimize.
- Analyze the optimal portfolio, including expected return, volatility, and Sharpe ratio.
```python
#Calculating log-returns
log_returns = np.log(adj_close_df / adj_close_df.shift(1))
#Taking care of missing values
log_returns = log_returns.dropna()

#Calculating the Co-Variance Matrix
cov_matrix = log_returns.cov()*252
print(cov_matrix)
```
> Defining The Portfolio Performance Matrix
```python
#Calculate the Portfolio Standard Deviation
def standard_deviation(weights, cov_matrix):
    variance = weights.T @ cov_matrix @ weights
    return np.sqrt(variance)

#Calculate the Expected Return of Portfolio (Assumption: Expected Returns are Based on Historical Returns for simplification purpose)
def expected_return(weights, log_returns):
    return np.sum(log_returns.mean()*weights)*252

#Calculate the Sharp Ratio Matrix (The Sharpe ratio is a measure of risk-adjusted return, calculated as: Sharpe Ratio = (Portfolio Return - Risk Free Rate (Rf)) / Standard Deviation of Portfolio Returns. It represents how much excess return you are receiving for the extra volatility you endure for holding a riskier asset.)
def sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate):
    return (expected_return(weights, log_returns) - risk_free_rate) / standard_deviation(weights, cov_matrix)

#To get the risk-free rate we use FredAPI
!pip install fredapi
from fredapi import Fred
```
> Portfolio Optimization
```python
#Calculate the Risk-Free Rate
fred = Fred(api_key='778e0a0b2a1dc1f46d0979a2f79ef915')
ten_year_treasury_rate = fred.get_series_latest_release('GS10') / 100

# Setting up as Risk_Free Rate
rf = ten_year_treasury_rate.iloc[-1]
print(rf)
#Define the function to minimize the Negative Sharp Ratio (In the case of scipy.optimize.minimize() function, there is no direct method to find the maximum value of a function.)
def neg_sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate):
    return -sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate)

#Set the Constraints Bound
constraints = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
bounds = [(0, 0.5) for _ in range(len(tickers))]
#Set the Initial Weights
initial_weights = np.array([1/len(tickers)]*len(tickers))
print(initial_weights)

#Optimize the Weights to Maximixe the Sharp Ratio
optimized_results = minimize(neg_sharpe_ratio, initial_weights, args=(log_returns, cov_matrix, rf), method='SLSQP', constraints=constraints, bounds=bounds)
#Get the optimal weights
optimal_weights = optimized_results.x
```
> Visualize the Optimal Portfolio Analytics
```python
#Optimal Portfolio Analytics
print("Optimal Weights:")
for ticker, weight in zip(tickers, optimal_weights):
    print(f"{ticker}: {weight:.4f}")

optimal_portfolio_return = expected_return(optimal_weights, log_returns)
optimal_portfolio_volatility = standard_deviation(optimal_weights, cov_matrix)
optimal_sharpe_ratio = sharpe_ratio(optimal_weights, log_returns, cov_matrix, rf)

print(f"Expected Annual Return: {optimal_portfolio_return:.4f}")
print(f"Expected Volatility: {optimal_portfolio_volatility:.4f}")
print(f"Sharpe Ratio: {optimal_sharpe_ratio:.4f}")
```
> Optimal Portfolio Plotting
```python
plt.figure(figsize=(10, 6))
plt.bar(tickers, optimal_weights)

plt.xlabel('Assets')
plt.ylabel('Optimal Weights')
plt.title('Optimal Portfolio Weights')

plt.show()
```
**3. Risk Measurement:**
- Perform Monte Carlo simulations (20,000 runs) to estimate portfolio gains and losses.
- Calculate the VaR at a 95% confidence level over a 5-day period.

> Creating an Equally Weighted Portfolio & Finding a Total Portfolio Expected Return and Standard Deviation
```python
#Assuming portfolio estimates total = $2million and we take equal weights for 7 assets = 14.29%
portfolio_value = 2000000
weights = np.array([1/len(tickers)]*len(tickers))
portfolio_expected_return = expected_return(weights, log_returns)
portfolio_std_dev = standard_deviation (weights, cov_matrix)
```
> Defining Functions for Monte Carlo Simulaton
```python
#Creating a Function that gives a random Z-Score based on Normal Distribution
def random_z_score():
    return np.random.normal(0, 1)

#Creating a Function to Calculate Scenario of Gain & Loss
days = 5

def scenario_gain_loss(portfolio_value, portfolio_std_dev, z_score, days):
    return portfolio_value * portfolio_expected_return * days + portfolio_value * portfolio_std_dev * z_score * np.sqrt(days)
     
#Running 20000 Monte carlo Simulations
simulations = 20000
scenarioReturn = []

for i in range(simulations):
    z_score = random_z_score()
    scenarioReturn.append(scenario_gain_loss(portfolio_value, portfolio_std_dev, z_score, days))
```
> Calculating Value at Risk (VaR) by Specifying a Confidence Interval
```python
confidence_interval = 0.95
VaR = -np.percentile(scenarioReturn, 100 * (1 - confidence_interval))
print(VaR)
```

> Plotting the scenarios generated by Monte Carlo Simulation - (Histogram of simulated portfolio returns with VaR indicated)
```python
plt.hist(scenarioReturn, bins=50, density=True)
plt.xlabel('Scenario Gain/Loss ($)')
plt.ylabel('Frequency')
plt.title(f'Distribution of Portfolio Gain/Loss Over {days} Days')
plt.axvline(-VaR, color='r', linestyle='dashed', linewidth=2, label=f'VaR at {confidence_interval:.0%} confidence level')
plt.legend()
plt.show()
```

## Results

**Optimal Portfolio:**

- Major allocations: 50% in BND and 46.23% in MSCI.
- Expected Return: 16.08%
- Expected Volatility: 9.74%
- Sharpe Ratio: 1.215
  
**Value at Risk of multi-asset portfolio:**
- VaR (95% confidence): $69,744.52 over a 5-day period
  
## Conclusion

In conclusion, the multi-asset portfolio optimization reveals a strategy that balances risk and return, primarily allocating investments to BND (50%) and MSCI (46.23%), with a small allocation to GLD (3.77%) for inflation hedging. The absence of UKX, SPY, QQQ, and VTI suggests these assets either fail to enhance risk-adjusted returns or add unnecessary volatility. The portfolio demonstrates strong performance metrics, with an expected annual return of 16.08%, moderate volatility of 9.74%, and a Sharpe ratio of 1.215, indicating favorable risk-adjusted returns.

Additionally, the Monte Carlo simulation-based Value at Risk (VaR) of $69,744.52 at a 95% confidence level over a 5-day period highlights the portfolio’s downside risk, implying a 5% chance that losses could exceed this amount during the period. Overall, this portfolio offers a strong return potential with a well-balanced risk profile. helping in making informed investment decisions.

## Author

Debolina Dutta

LinkedIn: (https://www.linkedin.com/in/duttadebolina/)
