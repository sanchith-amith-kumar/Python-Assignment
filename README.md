An Object-Oriented Approach to Data Matching and Visualization Using SQLAlchemy, Pandas, and Bokeh
Repository for the Programming with Python (DLMDSPWP01) written assignment at IU International University of Applied Sciences[cite: 1].

📌 Project Overview
This project implements a fully automated, robust data analysis pipeline using object-oriented programming (OOP) principles in Python[cite: 4]. The system ingests experimental training datasets, an unmapped testing dataset, and a reference file containing fifty ideal mathematical functions to perform optimal curve fitting, error boundary calculation, and interactive web visualization[cite: 4, 30].

Key Features:
Object-Oriented Architecture: Modular design featuring base and subclass data loaders (DataLoader, TrainingDataLoader, IdealFunctionLoader, TestDataLoader), a database manager, calculation engines, and visualizers[cite: 6, 13, 14].

Relational Database Persistence: Automatically provisions an SQLite database and stores structured tables via SQLAlchemy Object-Relational Mapping (ORM)[cite: 5, 14, 31].

Least-Squares Optimization: Parallelized NumPy routines to compute sum of squared errors (SSE) across all 50 reference functions[cite: 5, 30].

Validation & Error Mapping: Filters test coordinates against a localized error boundary defined by the maximum training deviation multiplied by square root of 2[cite: 6, 30].

Interactive Dashboards: Exports high-performance, standalone HTML visualization dashboards using the Bokeh library[cite: 7, 31].

Comprehensive Testing Suite: Includes built-in unittest coverage for data parsing, database roundtrips, and exception handling[cite: 18, 31].

🗂️ Project Structure
Python-Assignment/
│
├── dataset/
│   ├── train.csv         # Experimental training data (x, y1-y4)
│   ├── ideal.csv         # 50 Ideal reference functions (x, y1-y50)
│   └── test.csv          # Unmapped test data points (x, y)
│
├── results.db            # Generated SQLite database (persistence layer)
├── visualization.html    # Exported Bokeh interactive dashboard
├── main.py               # Main application entry point & test suite
└── README.md             # Project documentation

🚀 Getting Started & Installation
Prerequisites
Ensure you have Python 3.8+ installed along with the required scientific and database libraries:

pip install pandas numpy sqlalchemy bokeh

Running the Pipeline
Clone the repository and execute the main pipeline script[cite: 33]:

git clone https://github.com/sanchith-amith-kumar/Python-Assignment.git
cd Python-Assignment
python main.py

Running Unit Tests
To execute the built-in unit test suite and verify system integrity:

python main.py --test

📊 Database Schema Summary
The application automatically creates and populates the following relational tables inside results.db[cite: 14]:

training_data: Stores experimental inputs (x, y1, y2, y3, y4)[cite: 14, 31].

ideal_functions: Stores reference curves (x, y1 through y50)[cite: 14, 32].

test_results: Stores successfully mapped validation points matching Table 3 specification (X (test func), Y (test func), Delta Y (test func), No. of ideal func)[cite: 14, 32].

🌿 Git Workflow (Development Branch Strategy)
To contribute changes or add features to the development branch[cite: 33]:

1. Clone the develop branch
git clone --branch develop https://github.com/sanchith-amith-kumar/Python-Assignment.git 
cd

2. Create a feature branch
git checkout -b feature/new-feature develop

3. Stage, commit, and push your changes
git add .
git commit -m "feat: describe your feature here"
git push origin feature/new-feature

👤 Author
Sanchith Amith Kumar

Matriculation Number: 4252527[cite: 1]

Course: DLMDSPWP01 - Programming with Python | IU International University of Applied Sciences[cite: 1]
