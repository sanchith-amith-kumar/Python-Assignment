# Object-Oriented Data Matching & Web Visualization Pipeline

An end-to-end, production-grade data analysis application engineered using strict Object-Oriented Programming (OOP) patterns in Python. Built for the course **Programming with Python (DLMDSPWP01)**, this pipeline automates processing multi-dimensional data arrays, isolates mathematical reference curves through optimization constraints, and compiles modern web layouts.

---

## Key Features

- **Decoupled Architecture:** Enforces the Single Responsibility Principle across dedicated operational classes (`DataLoader`, `DatabaseManager`, `FunctionSelector`, `TestMapper`, and `Visualizer`).
- **Relational Integrity:** Implements SQLAlchemy Object-Relational Mapping (ORM) routines to securely initialize, map, and query data from a local SQLite environment.
- **Mathematical Fitting Optimization:** Uses an exhaustive least-squares approach via NumPy vectorization to map noisy experimental data to optimal reference functions.
- **Strict Validation Filtering:** Evaluates point compliance using a deviation threshold defined as `max(|Δy|) × √2`.
- **Interactive Web Dashboards:** Generates interactive HTML visualizations using Bokeh with zooming, hovering tooltips, and dynamic legend filtering.
- **Automated Validation Layer:** Includes `unittest`-based test cases to ensure correctness of processing logic.

---

## Project Structure

your-repository-name/  
├── dataset/                  # Raw CSV input files  
│   ├── train.csv  
│   ├── ideal.csv  
│   └── test.csv  
├── main.py                   # Main application script  
└── README.md                 # Project documentation  

---

## Prerequisites & Installation

Python 3.8+ is required.

Install dependencies:

pip install pandas numpy sqlalchemy bokeh

---

## Step-by-Step Execution

### Run Full Data Pipeline

python main.py

This will:
- Load datasets
- Process and map test data
- Store results in SQLite database
- Generate interactive visualization dashboard

Output files:
- results.db
- visualization.html

---

### Run Unit Tests

python main.py --test

Runs the internal validation suite and prints detailed test logs in the terminal.

---

## Component Architecture Overview

- **DataLoader:** Handles file input/output and CSV validation.
- **DatabaseManager:** Manages SQLite database interactions using ORM.
- **FunctionSelector:** Computes Sum of Squared Errors (SSE) to identify best-fit functions.
- **TestMapper:** Filters and validates test data using threshold-based logic.
- **Visualizer:** Builds interactive Bokeh HTML dashboards.

---

## Submission Information

Course Name: Programming with Python  
Course Code: DLMDSPWP01  
Submission Date: June 25, 2026
