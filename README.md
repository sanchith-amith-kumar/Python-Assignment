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
