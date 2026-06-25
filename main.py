"""
DLMDSPWP01 - Programming with Python
Written Assignment: Ideal Function Matching via Least-Squares

This program:
1. Loads training data (4 functions) and 50 ideal functions from CSV into SQLite via SQLAlchemy
2. Selects the 4 ideal functions that best fit the training data (least-squares criterion)
3. Maps test data points to the chosen ideal functions if deviation <= max_train_dev * sqrt(2)
4. Saves all results to SQLite
5. Visualizes training data, ideal functions, and test mappings using Bokeh

Author: [Student Name]
Matriculation Number: [Number]
Course: DLMDSPWP01 - Programming with Python
Date: 2025
"""

import os
import math
import sqlite3
import unittest
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, Column, Float, Integer, String, MetaData, Table, text
from sqlalchemy.orm import declarative_base, Session
from bokeh.plotting import figure, output_file, save
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource, Legend, LegendItem
from bokeh.palettes import Category10


# ──────────────────────────────────────────────
# Custom Exceptions
# ──────────────────────────────────────────────

class DataLoadError(Exception):
    """Raised when a CSV file cannot be loaded or has unexpected structure."""
    pass


class DatabaseError(Exception):
    """Raised when a database operation fails."""
    pass


class MappingError(Exception):
    """Raised when test-data mapping encounters an unexpected condition."""
    pass


# ──────────────────────────────────────────────
# Base ORM class
# ──────────────────────────────────────────────

Base = declarative_base()


# ──────────────────────────────────────────────
# Data Loading – Base class + subclasses
# ──────────────────────────────────────────────

class DataLoader:
    """
    Base class for loading CSV data into a pandas DataFrame.

    Attributes
    ----------
    filepath : str
        Path to the CSV file.
    """

    def __init__(self, filepath: str):
        """
        Parameters
        ----------
        filepath : str
            Path to the CSV file to load.
        """
        self.filepath = filepath
        self._df: pd.DataFrame | None = None

    def load(self) -> pd.DataFrame:
        """
        Load the CSV file into a DataFrame.

        Returns
        -------
        pd.DataFrame
            Loaded data.

        Raises
        ------
        DataLoadError
            If the file does not exist or cannot be parsed.
        """
        if not os.path.exists(self.filepath):
            raise DataLoadError(f"File not found: {self.filepath}")
        try:
            self._df = pd.read_csv(self.filepath)
        except Exception as exc:
            raise DataLoadError(f"Failed to parse {self.filepath}: {exc}") from exc
        return self._df

    @property
    def dataframe(self) -> pd.DataFrame:
        """Return the loaded DataFrame, loading it first if necessary."""
        if self._df is None:
            self.load()
        return self._df


class TrainingDataLoader(DataLoader):
    """
    Loads training data CSV (columns: x, y1, y2, y3, y4).

    Inherits from DataLoader.
    """

    EXPECTED_COLUMNS = {"x", "y1", "y2", "y3", "y4"}

    def load(self) -> pd.DataFrame:
        """
        Load and validate training data.

        Returns
        -------
        pd.DataFrame

        Raises
        ------
        DataLoadError
            If expected columns are missing.
        """
        df = super().load()
        missing = self.EXPECTED_COLUMNS - set(df.columns)
        if missing:
            raise DataLoadError(f"Training CSV missing columns: {missing}")
        return df


class IdealFunctionLoader(DataLoader):
    """
    Loads ideal functions CSV (columns: x, y1 … y50).

    Inherits from DataLoader.
    """

    def load(self) -> pd.DataFrame:
        """
        Load and validate ideal functions data.

        Returns
        -------
        pd.DataFrame

        Raises
        ------
        DataLoadError
            If x column is missing or fewer than 50 y-columns are present.
        """
        df = super().load()
        if "x" not in df.columns:
            raise DataLoadError("Ideal functions CSV missing 'x' column.")
        y_cols = [c for c in df.columns if c != "x"]
        if len(y_cols) < 50:
            raise DataLoadError(
                f"Expected 50 ideal function columns, found {len(y_cols)}."
            )
        return df


class TestDataLoader(DataLoader):
    """
    Loads test data CSV (columns: x, y).

    Inherits from DataLoader.
    """

    EXPECTED_COLUMNS = {"x", "y"}

    def load(self) -> pd.DataFrame:
        """
        Load and validate test data.

        Returns
        -------
        pd.DataFrame

        Raises
        ------
        DataLoadError
            If expected columns are missing.
        """
        df = super().load()
        missing = self.EXPECTED_COLUMNS - set(df.columns)
        if missing:
            raise DataLoadError(f"Test CSV missing columns: {missing}")
        return df


# ──────────────────────────────────────────────
# Database Manager
# ──────────────────────────────────────────────

class DatabaseManager:
    """
    Manages all SQLite database operations via SQLAlchemy.

    Creates and populates three tables:
    - training_data   : x, y1, y2, y3, y4
    - ideal_functions : x, y1 … y50
    - test_results    : x, y, delta_y, ideal_function_no
    """

    def __init__(self, db_path: str = "results.db"):
        """
        Parameters
        ----------
        db_path : str
            File path for the SQLite database.
        """
        self.db_path = db_path
        try:
            self.engine = create_engine(f"sqlite:///{db_path}", echo=False)
            self.metadata = MetaData()
        except Exception as exc:
            raise DatabaseError(f"Could not create database engine: {exc}") from exc

    def write_dataframe(self, df: pd.DataFrame, table_name: str) -> None:
        """
        Write a DataFrame to a database table (replaces if exists).

        Parameters
        ----------
        df : pd.DataFrame
            Data to write.
        table_name : str
            Target table name.

        Raises
        ------
        DatabaseError
            If the write operation fails.
        """
        try:
            df.to_sql(table_name, con=self.engine, if_exists="replace", index=False)
        except Exception as exc:
            raise DatabaseError(
                f"Failed to write table '{table_name}': {exc}"
            ) from exc

    def write_test_results(self, results: list[dict]) -> None:
        """
        Write test mapping results to the 'test_results' table.

        Parameters
        ----------
        results : list[dict]
            Each dict must have keys: x, y, delta_y, ideal_function_no.

        Raises
        ------
        DatabaseError
            If the write operation fails.
        """
        if not results:
            return
        df = pd.DataFrame(results)
        self.write_dataframe(df, "test_results")

    def read_table(self, table_name: str) -> pd.DataFrame:
        """
        Read a full table back as a DataFrame.

        Parameters
        ----------
        table_name : str

        Returns
        -------
        pd.DataFrame
        """
        try:
            with self.engine.connect() as conn:
                return pd.read_sql(f"SELECT * FROM {table_name}", conn)
        except Exception as exc:
            raise DatabaseError(f"Failed to read table '{table_name}': {exc}") from exc


# ──────────────────────────────────────────────
# Function Selector – picks best 4 ideal functions
# ──────────────────────────────────────────────

class FunctionSelector:
    """
    Selects the four ideal functions (from fifty) that best fit the four
    training functions using the least-squares criterion.
    """

    def __init__(self, train_df: pd.DataFrame, ideal_df: pd.DataFrame):
        """
        Parameters
        ----------
        train_df : pd.DataFrame
            Training data with columns x, y1, y2, y3, y4.
        ideal_df : pd.DataFrame
            Ideal functions data with columns x, y1 … y50.
        """
        self.train_df = train_df.sort_values("x").reset_index(drop=True)
        self.ideal_df = ideal_df.sort_values("x").reset_index(drop=True)
        self.ideal_cols = [c for c in ideal_df.columns if c != "x"]
        self.train_cols = ["y1", "y2", "y3", "y4"]

        # Results populated by select()
        self.chosen: dict[str, str] = {}      # train_col -> ideal_col
        self.max_deviations: dict[str, float] = {}  # train_col -> max |y_train - y_ideal|

    def _sum_of_squares(self, train_col: str, ideal_col: str) -> float:
        """
        Compute sum of squared differences between a training column and an ideal column.

        Parameters
        ----------
        train_col : str
        ideal_col : str

        Returns
        -------
        float
        """
        diff = self.train_df[train_col].values - self.ideal_df[ideal_col].values
        return float(np.sum(diff ** 2))

    def select(self) -> dict[str, str]:
        """
        For each training function, find the ideal function with minimum SSE.

        Returns
        -------
        dict[str, str]
            Mapping of training column name -> best ideal column name.
        """
        for tc in self.train_cols:
            best_col = min(
                self.ideal_cols,
                key=lambda ic: self._sum_of_squares(tc, ic)
            )
            self.chosen[tc] = best_col
            # Compute max absolute deviation for threshold use later
            diffs = np.abs(
                self.train_df[tc].values - self.ideal_df[best_col].values
            )
            self.max_deviations[tc] = float(np.max(diffs))

        return self.chosen


# ──────────────────────────────────────────────
# Test Data Mapper
# ──────────────────────────────────────────────

class TestMapper:
    """
    Maps each test (x, y) point to one of the four chosen ideal functions,
    provided the deviation does not exceed max_train_deviation * sqrt(2).
    """

    SQRT2 = math.sqrt(2)

    def __init__(
        self,
        test_df: pd.DataFrame,
        ideal_df: pd.DataFrame,
        chosen: dict[str, str],
        max_deviations: dict[str, float],
    ):
        """
        Parameters
        ----------
        test_df : pd.DataFrame
            Test data with columns x, y.
        ideal_df : pd.DataFrame
            Full ideal functions dataset.
        chosen : dict[str, str]
            Mapping train_col -> ideal_col from FunctionSelector.
        max_deviations : dict[str, float]
            Per-training-function maximum deviation from FunctionSelector.
        """
        self.test_df = test_df
        self.ideal_df = ideal_df.set_index("x")
        self.chosen = chosen          # {train_col: ideal_col}
        self.max_deviations = max_deviations
        self.results: list[dict] = []

    def map(self) -> list[dict]:
        """
        Perform the mapping of test points to ideal functions.

        For each test point:
        - Look up the ideal function value at the test x (exact match required).
        - Compute deviation |y_test - y_ideal|.
        - Accept if deviation <= max_train_deviation[train_col] * sqrt(2).
        - Among all accepting ideal functions, choose the one with smallest deviation.

        Returns
        -------
        list[dict]
            List of result dicts: {x, y, delta_y, ideal_function_no}
            Only points that matched at least one ideal function are included.

        Raises
        ------
        MappingError
            If an unexpected structural error occurs.
        """
        self.results = []

        for _, row in self.test_df.iterrows():
            x_val = row["x"]
            y_val = row["y"]

            best_match = None
            best_dev = float("inf")

            for train_col, ideal_col in self.chosen.items():
                threshold = self.max_deviations[train_col] * self.SQRT2

                # Look up ideal function value at this x
                if x_val not in self.ideal_df.index:
                    continue  # No matching x in ideal dataset

                y_ideal = self.ideal_df.loc[x_val, ideal_col]
                deviation = abs(y_val - y_ideal)

                if deviation <= threshold and deviation < best_dev:
                    best_dev = deviation
                    best_match = {
                        "x": x_val,
                        "y": y_val,
                        "delta_y": round(deviation, 6),
                        "ideal_function_no": ideal_col,
                    }

            if best_match is not None:
                self.results.append(best_match)

        return self.results


# ──────────────────────────────────────────────
# Visualizer
# ──────────────────────────────────────────────

class Visualizer:
    """
    Creates Bokeh HTML visualizations for training data, chosen ideal functions,
    and test data mappings.
    """

    COLORS = Category10[4]

    def __init__(
        self,
        train_df: pd.DataFrame,
        ideal_df: pd.DataFrame,
        test_df: pd.DataFrame,
        chosen: dict[str, str],
        test_results: list[dict],
        output_path: str = "visualization.html",
    ):
        """
        Parameters
        ----------
        train_df : pd.DataFrame
        ideal_df : pd.DataFrame
        test_df : pd.DataFrame
        chosen : dict[str, str]
        test_results : list[dict]
        output_path : str
        """
        self.train_df = train_df
        self.ideal_df = ideal_df
        self.test_df = test_df
        self.chosen = chosen
        self.test_results = pd.DataFrame(test_results) if test_results else pd.DataFrame()
        self.output_path = output_path
        self.train_cols = ["y1", "y2", "y3", "y4"]

    def _make_training_plot(self, idx: int, train_col: str, ideal_col: str) -> figure:
        """
        Build a single Bokeh figure comparing one training function to its best ideal match.

        Parameters
        ----------
        idx : int
            Plot index (for color selection).
        train_col : str
        ideal_col : str

        Returns
        -------
        bokeh.plotting.figure
        """
        color = self.COLORS[idx]
        p = figure(
            title=f"{train_col} vs Ideal {ideal_col}",
            x_axis_label="x",
            y_axis_label="y",
            width=500,
            height=350,
            tools="pan,wheel_zoom,reset,save",
        )
        p.line(
            self.train_df["x"],
            self.train_df[train_col],
            line_color=color,
            line_width=2,
            legend_label=f"Training {train_col}",
        )
        p.line(
            self.ideal_df["x"],
            self.ideal_df[ideal_col],
            line_color="black",
            line_width=1,
            line_dash="dashed",
            legend_label=f"Ideal {ideal_col}",
        )
        p.legend.location = "top_left"
        return p

    def _make_test_plot(self) -> figure:
        """
        Build a Bokeh figure showing all chosen ideal functions and mapped test points.

        Returns
        -------
        bokeh.plotting.figure
        """
        p = figure(
            title="Test Data Mapping to Chosen Ideal Functions",
            x_axis_label="x",
            y_axis_label="y",
            width=1000,
            height=400,
            tools="pan,wheel_zoom,reset,save",
        )

        # Plot chosen ideal functions
        for idx, (train_col, ideal_col) in enumerate(self.chosen.items()):
            p.line(
                self.ideal_df["x"],
                self.ideal_df[ideal_col],
                line_color=self.COLORS[idx],
                line_width=2,
                legend_label=f"Ideal {ideal_col} (best for {train_col})",
            )

        # Plot mapped test points
        if not self.test_results.empty:
            p.circle(
                self.test_results["x"],
                self.test_results["y"],
                size=8,
                color="red",
                legend_label="Mapped test points",
                alpha=0.8,
            )

        # Plot unmatched test points
        if not self.test_results.empty:
            matched_x = set(self.test_results["x"])
        else:
            matched_x = set()
        unmatched = self.test_df[~self.test_df["x"].isin(matched_x)]
        if not unmatched.empty:
            p.x(
                unmatched["x"],
                unmatched["y"],
                size=8,
                color="gray",
                legend_label="Unmatched test points",
            )

        p.legend.location = "top_left"
        return p

    def render(self) -> None:
        """
        Generate and save the full Bokeh HTML output.
        """
        output_file(self.output_path, title="DLMDSPWP01 Visualizations")

        plots = []
        for idx, (train_col, ideal_col) in enumerate(self.chosen.items()):
            plots.append(self._make_training_plot(idx, train_col, ideal_col))

        grid = gridplot([plots[:2], plots[2:]])
        test_plot = self._make_test_plot()

        from bokeh.layouts import column as bk_column
        layout = bk_column(grid, test_plot)
        save(layout)
        print(f"Visualization saved to: {self.output_path}")


# ──────────────────────────────────────────────
# Pipeline – orchestrates everything
# ──────────────────────────────────────────────

class Pipeline:
    """
    Orchestrates the full data processing pipeline:
    load → store → select → map → store results → visualize.
    """

    def __init__(
        self,
        train_path: str = "dataset/train.csv",
        ideal_path: str = "dataset/ideal.csv",
        test_path: str = "dataset/test.csv",
        db_path: str = "results.db",
        viz_path: str = "visualization.html",
    ):
        """
        Parameters
        ----------
        train_path : str
        ideal_path : str
        test_path : str
        db_path : str
        viz_path : str
        """
        self.train_path = train_path
        self.ideal_path = ideal_path
        self.test_path = test_path
        self.db_path = db_path
        self.viz_path = viz_path

    def run(self) -> None:
        """
        Execute the complete pipeline end-to-end.
        """
        print("=== DLMDSPWP01 Pipeline Starting ===\n")

        # 1. Load data
        print("[1/5] Loading CSV data...")
        train_df = TrainingDataLoader(self.train_path).load()
        ideal_df = IdealFunctionLoader(self.ideal_path).load()
        test_df = TestDataLoader(self.test_path).load()
        print(f"  Training:  {len(train_df)} rows")
        print(f"  Ideal:     {len(ideal_df)} rows, {len(ideal_df.columns)-1} functions")
        print(f"  Test:      {len(test_df)} rows")

        # 2. Store training + ideal in SQLite
        print("\n[2/5] Writing to SQLite database...")
        db = DatabaseManager(self.db_path)
        db.write_dataframe(train_df, "training_data")
        db.write_dataframe(ideal_df, "ideal_functions")
        print(f"  Database: {self.db_path}")

        # 3. Select best 4 ideal functions
        print("\n[3/5] Selecting best ideal functions (Least-Squares)...")
        selector = FunctionSelector(train_df, ideal_df)
        chosen = selector.select()
        for tc, ic in chosen.items():
            sse = selector._sum_of_squares(tc, ic)
            max_dev = selector.max_deviations[tc]
            print(f"  {tc} -> {ic}  |  SSE={sse:.4f}  |  max_dev={max_dev:.6f}")

        # 4. Map test data
        print("\n[4/5] Mapping test data to chosen ideal functions...")
        mapper = TestMapper(test_df, ideal_df, chosen, selector.max_deviations)
        results = mapper.map()
        print(f"  Mapped {len(results)} / {len(test_df)} test points")

        # Store results
        db.write_test_results(results)
        print("  Test results written to database.")

        # 5. Visualize
        print("\n[5/5] Generating visualizations...")
        viz = Visualizer(train_df, ideal_df, test_df, chosen, results, self.viz_path)
        viz.render()

        print("\n=== Pipeline Complete ===")
        print(f"  Database:      {self.db_path}")
        print(f"  Visualization: {self.viz_path}")


# ──────────────────────────────────────────────
# Unit Tests
# ──────────────────────────────────────────────

class TestDataLoaders(unittest.TestCase):
    """Unit tests for DataLoader subclasses."""

    def test_training_loader_missing_file(self):
        """DataLoadError raised for non-existent file."""
        with self.assertRaises(DataLoadError):
            TrainingDataLoader("no_such_file.csv").load()

    def test_training_loader_columns(self):
        """TrainingDataLoader validates required columns."""
        import tempfile, csv
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            writer = csv.writer(f)
            writer.writerow(["x", "y1", "y2"])  # missing y3, y4
            writer.writerow([0, 1, 2])
            name = f.name
        with self.assertRaises(DataLoadError):
            TrainingDataLoader(name).load()
        os.unlink(name)

    def test_test_loader_columns(self):
        """TestDataLoader validates required columns."""
        import tempfile, csv
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            writer = csv.writer(f)
            writer.writerow(["x"])  # missing y
            writer.writerow([0])
            name = f.name
        with self.assertRaises(DataLoadError):
            TestDataLoader(name).load()
        os.unlink(name)


class TestFunctionSelector(unittest.TestCase):
    """Unit tests for FunctionSelector."""

    def _make_dfs(self):
        x = np.linspace(-5, 5, 11)
        train_df = pd.DataFrame({
            "x": x,
            "y1": np.sin(x),
            "y2": np.cos(x),
            "y3": x ** 2,
            "y4": x ** 3,
        })
        ideal_data = {"x": x}
        # y1 matches sin, y2 matches cos, y3 matches x^2, y4 matches x^3
        ideal_data["y1"] = np.sin(x)
        ideal_data["y2"] = np.cos(x)
        ideal_data["y3"] = x ** 2
        ideal_data["y4"] = x ** 3
        # Add noise decoys
        for i in range(5, 51):
            ideal_data[f"y{i}"] = np.random.randn(len(x)) * 100
        ideal_df = pd.DataFrame(ideal_data)
        return train_df, ideal_df

    def test_select_returns_four(self):
        """FunctionSelector.select() returns exactly 4 mappings."""
        train_df, ideal_df = self._make_dfs()
        sel = FunctionSelector(train_df, ideal_df)
        chosen = sel.select()
        self.assertEqual(len(chosen), 4)

    def test_select_perfect_match(self):
        """FunctionSelector finds correct ideal columns for perfect data."""
        train_df, ideal_df = self._make_dfs()
        sel = FunctionSelector(train_df, ideal_df)
        chosen = sel.select()
        self.assertEqual(chosen["y1"], "y1")
        self.assertEqual(chosen["y2"], "y2")
        self.assertEqual(chosen["y3"], "y3")
        self.assertEqual(chosen["y4"], "y4")

    def test_max_deviations_zero_for_perfect(self):
        """Max deviations are zero when ideal matches training exactly."""
        train_df, ideal_df = self._make_dfs()
        sel = FunctionSelector(train_df, ideal_df)
        sel.select()
        for tc in ["y1", "y2", "y3", "y4"]:
            self.assertAlmostEqual(sel.max_deviations[tc], 0.0, places=10)


class TestTestMapper(unittest.TestCase):
    """Unit tests for TestMapper."""

    def _make_data(self):
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        train_df = pd.DataFrame({"x": x, "y1": x, "y2": -x, "y3": x**2, "y4": x**3})
        ideal_data = {"x": x, "y1": x, "y2": -x, "y3": x**2, "y4": x**3}
        for i in range(5, 51):
            ideal_data[f"y{i}"] = np.zeros(len(x))
        ideal_df = pd.DataFrame(ideal_data)
        chosen = {"y1": "y1", "y2": "y2", "y3": "y3", "y4": "y4"}
        max_devs = {"y1": 0.0, "y2": 0.0, "y3": 0.0, "y4": 0.0}
        return train_df, ideal_df, chosen, max_devs

    def test_perfect_match_mapped(self):
        """Test point exactly on ideal function maps successfully."""
        train_df, ideal_df, chosen, max_devs = self._make_data()
        test_df = pd.DataFrame({"x": [1.0], "y": [1.0]})
        # y1 at x=1 is 1.0; deviation=0 <= 0*sqrt(2)=0 → should map
        mapper = TestMapper(test_df, ideal_df, chosen, max_devs)
        results = mapper.map()
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["ideal_function_no"], "y1")

    def test_exceeding_threshold_not_mapped(self):
        """Test point beyond sqrt(2)*max_dev threshold is not mapped."""
        train_df, ideal_df, chosen, max_devs = self._make_data()
        # max_devs are 0, threshold=0; any deviation > 0 → not mapped
        test_df = pd.DataFrame({"x": [1.0], "y": [99.0]})
        mapper = TestMapper(test_df, ideal_df, chosen, max_devs)
        results = mapper.map()
        self.assertEqual(len(results), 0)

    def test_unknown_x_not_mapped(self):
        """Test point with x not in ideal dataset is not mapped."""
        train_df, ideal_df, chosen, max_devs = self._make_data()
        test_df = pd.DataFrame({"x": [999.0], "y": [0.0]})
        mapper = TestMapper(test_df, ideal_df, chosen, max_devs)
        results = mapper.map()
        self.assertEqual(len(results), 0)


class TestDatabaseManager(unittest.TestCase):
    """Unit tests for DatabaseManager."""

    def test_write_and_read_roundtrip(self):
        """Data written to DB can be read back correctly."""
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            db = DatabaseManager(db_path)
            df = pd.DataFrame({"x": [1.0, 2.0], "y1": [3.0, 4.0]})
            db.write_dataframe(df, "test_table")
            result = db.read_table("test_table")
            self.assertEqual(list(result["x"]), [1.0, 2.0])
            self.assertEqual(list(result["y1"]), [3.0, 4.0])
        finally:
            os.unlink(db_path)


# ────────────────────────────────────────────────────────────────────────────
# Version Control Documentation String & System Main Router Entry
# ────────────────────────────────────────────────────────────────────────────

# Using standard ASCII characters to prevent UnicodeEncodeError on Windows terminals
GIT_COMMANDS = """
# === Git Commands for Additional Task 1.3 ===================================
# 1. Clone the 'develop' branch from the remote repository to your local PC:
git clone --branch develop <remote-repository-url> <local-folder-name>

# 2. Navigate into the cloned project:
cd <local-folder-name>

# 3. Create a feature branch for your new function:
git checkout -b feature/new-function develop

# 4. Make your changes - e.g., add the new function to main.py, then stage:
git add .

# 5. Commit with a descriptive message:
git commit -m "feat: add new function for [describe purpose]"

# 6. Push your feature branch to the remote:
git push origin feature/new-function

# 7. On GitHub/GitLab: open a Pull Request from feature/new-function -> develop.
# ============================================================================
"""


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if "--test" in sys.argv:
        # Run unit tests
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        for cls in [
            TestDataLoaders,
            TestFunctionSelector,
            TestTestMapper,
            TestDatabaseManager,
        ]:
            suite.addTests(loader.loadTestsFromTestCase(cls))
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        sys.exit(0 if result.wasSuccessful() else 1)
    else:
        print(GIT_COMMANDS)
        pipeline = Pipeline(
            train_path="dataset/train.csv",
            ideal_path="dataset/ideal.csv",
            test_path="dataset/test.csv",
            db_path="results.db",
            viz_path="visualization.html",
        )
        pipeline.run()