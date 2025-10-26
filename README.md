
# Brent Oil Price Change Point Analysis

An intelligent analytics platform for detecting and analyzing significant shifts in Brent oil market dynamics using Bayesian statistical modeling.

## Table of Contents
- [Overview](#overview)
- [Problem](#problem)
- [Approach](#approach)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Impact](#impact)
- [Challenges](#challenges)
- [Next Steps](#next-steps)
- [Technologies](#technologies)
- [License](#license)

## Overview
This project analyzes historical Brent oil price data to identify **change points**, significant moments where price behavior shifts dramatically. These shifts often correspond to real-world events such as geopolitical conflicts, economic changes, or policy decisions. By detecting these points and linking them to events, the project quantifies their impact on oil prices.

## Problem
Oil prices are volatile and influenced by complex geopolitical and economic events. Identifying when and why prices shift is critical for traders, analysts, and policymakers.

## Approach
- **Bayesian Change Point Detection**: Built using PyMC and MCMC sampling to identify k change points in the time series.
- **Data Segmentation**: Divides price data into segments with their own mean behaviors but shared volatility.
- **Event Mapping**: Change points are correlated with historical events within a 90-day window.
- **Impact Analysis**: Measures mean price change, volatility, and percentage change around detected points.
- **Visualization**: Interactive graphs using Plotly and posterior probability plots using ArviZ.

## Features
- Detect multiple change points in Brent oil price time series.
- Map changes to historical geopolitical or economic events.
- Quantify impact on price and volatility.
- Interactive visualization of results with dynamic plots.
- Robust data preprocessing and log return calculation.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/brent-oil-change-point.git
   cd brent-oil-change-point


2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Prepare your data:

   * `brent_prices.csv` with daily prices and dates.
   * `events.csv` with historical events affecting oil markets.
2. Run preprocessing:

   ```bash
   python data_preprocessing.py
   ```
3. Run the Bayesian model:

   ```bash
   python change_point_model.py
   ```
4. Generate visualizations:

   ```bash
   python visualize_results.py
   ```

## Results

* Interactive plots showing Brent oil prices, change points, and mapped events.
* Posterior probability distributions for detected change points.
* Quantified metrics for mean price change, volatility, and percentage change.

## Impact

The platform enables stakeholders to:

* Understand historical market behavior.
* Identify periods of risk.
* Make data-driven decisions for trading and policy analysis.

## Challenges

* Sorting change points chronologically in PyMC and Aesara.
* Tensor conversion errors during calculations.
* Slow MCMC sampling for multiple change points.
* Change points without nearby events.

## Next Steps

* Allow segments to have individual volatility.
* Test hierarchical Bayesian models for flexible detection.
* Validate predictions with simulated data and traditional methods.
* Automate report generation and package with Docker.

## Technologies

* Python
* PyMC
* ArviZ
* Plotly
* Pandas & NumPy

## License

This project is licensed under the MIT License.


