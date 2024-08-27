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

** 1. Data Collection:**

- Download historical data for selected tickers (UKX, SPY, GLD, QQQ, BND, MSCI, VTI) using yfinance.
- Time range: Last 10 years.

**2. Portfolio Optimization:**
- Calculate log returns and covariance matrix of the assets.
- Optimize the portfolio to maximize the Sharpe ratio using scipy.optimize.minimize.
- Analyze the optimal portfolio, including expected return, volatility, and Sharpe ratio.

**3. Risk Measurement:**
- Perform Monte Carlo simulations (20,000 runs) to estimate portfolio gains and losses.
- Calculate the VaR at a 95% confidence level over a 5-day period.
  
**4. Visualization:**
- Bar plot of optimal portfolio weights.
- Histogram of simulated portfolio returns with VaR indicated.

**Installation**
To run this project, you need the following Python libraries:

yfinance 
pandas 
numpy 
scipy 
matplotlib 
fredapi

## Usage
**1. Run Portfolio Optimization:**
- Calculate and display optimal asset weights, expected annual return, portfolio volatility, and Sharpe ratio.
**2. Monte Carlo Simulation for VaR:**
- Estimate the Value at Risk over a 5-day period at a 95% confidence level.

## Results

**Optimal Portfolio:**

- Major allocations: 50% in BND and 46.23% in MSCI.
- Expected Return: 16.08%
- Expected Volatility: 9.74%
- Sharpe Ratio: 1.215
- Value at Risk: VaR (95% confidence): $69,744.52 over a 5-day period
  
## Conclusion

The optimized portfolio strategy is conservative, focusing on bonds and global equities to minimize risk while achieving decent returns. The VaR analysis provides insights into potential losses, helping in making informed investment decisions.
