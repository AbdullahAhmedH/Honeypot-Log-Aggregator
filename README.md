# Honeypot Log Aggregator

## Overview
The Honeypot Log Aggregator is a GUI-based tool that processes and analyzes honeypot logs to classify attack types and visualize attack trends. It leverages an XGBoost model for classification and provides intuitive visualizations using Seaborn and Matplotlib.

## Features
- Import honeypot logs in `.txt` format
- Classify attack types using a trained XGBoost model
- Display attack trends with heatmaps and trend charts
- Intuitive PyQt5-based graphical user interface

## Installation
Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/AbdullahAhmedH/Honeypot-Log-Aggregator.git
cd Honeypot-Log-Aggregator
```

## Prerequisites
Ensure you have the following dependencies installed in python 3.9:

```bash
pip install -r requirements.txt
```

## Usage
1. Run the application:
   ```bash
   python gui2.py
   ```
2. Click `Import Log File` to select a honeypot log file.
3. Click the `>` button to process the logs.
4. View classified logs and select different visualizations (Heatmap/Trend Chart).

## Screenshots

