# Object-Oriented Data Matching & Web Visualization Pipeline

An end-to-end, production-grade data analysis application engineered using strict Object-Oriented Programming (OOP) patterns in Python. Built for the course **Programming with Python (DLMDSPWP01)**, this pipeline automates processing multi-dimensional data arrays, isolates mathematical reference curves through optimization constraints, and compiles modern web layouts.

---

## Key Features

- **Decoupled Architecture:** Enforces the Single Responsibility Principle across dedicated operational classes (`DataLoader`, `DatabaseManager`, `FunctionSelector`, `TestMapper`, and `Visualizer`).
- **Relational Integrity:** Implements SQLAlchemy Object-Relational Mapping (ORM) routines to securely initialize, map, and query data from a local transactional SQLite environment.
- **Mathematical Fitting Optimization:** Employs an exhaustive least-squares routine via NumPy vectorization to map noisy experimental lines back to global minimum profiles.
- **Strict Validation Filtering:** Evaluates point compliance across independent testing tracks using a locked deviation boundary multiplier ($\max(|\Delta y|) \times \sqrt{2}$).
- **Interactive Web Dashboards:** Generates HTML interface visualizations utilizing the Bokeh engine featuring active pans, tool hovers, and dynamic legend cross-filters.
- **Automated Validation Layer:** Contains a test execution tier structured using standard `unittest` blocks to confirm functional logic coverage.

---

## Prerequisites & Installation

To run the pipeline and verification suites locally, ensure you have Python 3.8+ installed along with the required third-party ecosystem libraries. 

Install the dependencies via your terminal:
```bash
pip install pandas numpy sqlalchemy bokeh

Step-by-Step Execution GuidelinesThe application main file (main.py) features a dual-purpose control entry interface driven by system terminal arguments:1. Run the Main Data PipelineTo execute the complete production data pipeline—which processes files, populates the relational SQLite database, maps test coordinates, and generates interactive graphs—run:Bashpython main.py
Expected Pipeline Outputs:results.db (SQLite relational snapshot repository containing tables: training_data, ideal_functions, and test_results)visualization.html (Standalone, interactive multi-panel vector plotting layout dashboard)2. Trigger the Automated Unit Testing SuiteTo verify the internal processing constraints and ensure file parsers or mathematical matrices are scaling without structural bugs, run:Bashpython main.py --test
This triggers the internal testing engine, running diagnostic profiles across all underlying class setups and printing a detailed verbosity execution log in the terminal window.Component Architecture OverviewDataLoader Sub-layer: Handles disk I/O operations, structural dimension parsing validations, and returns unified Pandas DataFrame containers.DatabaseManager Sub-layer: Abstracts row writing operations, handling transactional connections and safe execution parameters without raw string injections.FunctionSelector Sub-layer: Calculates cumulative Sum of Squared Errors ($SSE$) over variations using vectorized NumPy iterations to pick top matching mathematical curves.TestMapper Sub-layer: Performs individual coordinate evaluations, filtering structural noise anomalies using maximum threshold criteria.Visualizer Sub-layer: Groups separate plot layouts and binds underlying ColumnDataSource browser parameters to export an HTML analytics file.Submission MetainformationCourse Name: Programming with PythonCourse Code: DLMDSPWP01Submission Date: June 25, 2026





