# Book-Crossing Big Data Analytics

A Big Data Analytics project built on the Book-Crossing dataset. The project uses **Apache Spark / PySpark** for data preparation, quality assessment, exploratory analysis, recommendation modelling, model comparison, and scalability experiments.

> This repository is the Big Data Analytics project. The older Flask/PostgreSQL Book Management application is not the focus of this README.

## Project objectives

- Inspect and assess the quality of the Book-Crossing Books, Users, and Ratings datasets.
- Clean and transform the data with PySpark.
- Explore rating behaviour, book popularity, user activity, and publication-year patterns.
- Compare recommendation / rating-prediction approaches, including Spark MLlib ALS and Random Forest, against a global-mean baseline.
- Measure runtime behaviour at multiple data-volume levels.

## Technology stack

- Apache Spark 3.5.6 and PySpark
- Spark SQL and Spark MLlib
- Python
- Matplotlib for generated figures
- Java 11 (compatible Spark runtime)

## Repository structure

```text
.
├── analysis/
│   ├── clean_with_spark.py
│   ├── eda_with_spark.py
│   ├── extended_eda.py
│   ├── inspect_dataset.py
│   ├── quality_checks.py
│   ├── relationship_checks.py
│   ├── create_visualizations.py
│   └── results/
│       ├── data_quality_audit/
│       ├── fair_model_comparison/
│       ├── figures/
│       ├── als_recommender/
│       ├── als_tuning/
│       └── scalability_experiment_repeated/
├── models/
│   ├── als_recommender.py
│   ├── als_tuning.py
│   ├── random_forest_rating.py
│   ├── fair_model_comparison.py
│   ├── data_quality_audit.py
│   └── scalability_experiment_repeated.py
├── data/
│   ├── raw/          # Original CSV files
│   └── processed/    # Generated Parquet data (not committed)
├── requirements-bigdata.txt
└── README.md
```

## Dataset

The project uses the Book-Crossing dataset, including:

- `Books.csv`: book metadata.
- `Users.csv`: user attributes.
- `Ratings.csv`: user-book rating interactions.

Place the original files under `data/raw/`. Keep the original filenames and preserve the source data. Do not commit credentials or unrelated personal data.

The processed Parquet directory is excluded from Git. Generate it by running the preprocessing script before running analyses or models.

## Environment setup

Recommended environment:

- Python 3.11
- Java 17 (OpenJDK / Temurin)
- Apache Spark / PySpark 3.5.6

Create and activate a virtual environment, then install the Big Data dependencies:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-bigdata.txt
```

Linux / WSL:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-bigdata.txt
```

Check the runtime:

```bash
java -version
python --version
python -c "import pyspark; print(pyspark.__version__)"
```

## Run with Docker (recommended for a consistent environment)

The Spark environment is separate from the legacy Flask application. It uses `Dockerfile.spark` and `docker-compose.spark.yml`.

Build and start the Spark container from the repository root:

```powershell
docker compose -f docker-compose.spark.yml up -d --build
docker compose -f docker-compose.spark.yml ps
```

Run project commands inside the container (the repository is mounted at `/workspace`):

```powershell
docker compose -f docker-compose.spark.yml exec spark python analysis/inspect_dataset.py
docker compose -f docker-compose.spark.yml exec spark python analysis/clean_with_spark.py
```

After preprocessing succeeds, run the EDA and model commands listed below using the same `docker compose ... exec spark python ...` pattern. For example:

```powershell
docker compose -f docker-compose.spark.yml exec spark python models/fair_model_comparison.py
```

Stop the Spark container when finished:

```powershell
docker compose -f docker-compose.spark.yml down
```

## Run directly on Windows PowerShell

Install Java 17 and Python 3.11, then from the repository root:

```powershell
java -version
python --version
python -m venv .venv
.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-bigdata.txt
```

If PowerShell blocks environment activation, run this once in the current terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\\Scripts\\Activate.ps1
```

All analytics scripts now resolve data and output paths relative to their own file locations, so run them from the repository root in either environment.

## Execution order

Run commands from the repository root so the relative input/output paths resolve consistently.

1. Inspect the raw dataset:

```bash
python analysis/inspect_dataset.py
```

2. Clean and prepare the data (creates the processed Parquet dataset):

```bash
python analysis/clean_with_spark.py
```

3. Run data-quality checks and exploratory analyses:

```bash
python models/data_quality_audit.py
python analysis/quality_checks.py
python analysis/relationship_checks.py
python analysis/eda_with_spark.py
python analysis/extended_eda.py
```

4. Train and evaluate the models:

```bash
python models/als_recommender.py
python models/random_forest_rating.py
python models/fair_model_comparison.py
python models/als_tuning.py
```

5. Run the repeated scalability experiment (this may take longer):

```bash
python models/scalability_experiment_repeated.py
```

6. Generate figures from the saved analysis outputs:

```bash
python analysis/create_visualizations.py
```

Run each stage only after the previous stage has completed successfully. If a script reports that an input path is missing, confirm that you are in the repository root and that preprocessing generated the expected Parquet directory.

## Evaluation approach

The project reports regression metrics such as **RMSE** and **MAE**; lower values indicate lower prediction error. Model comparisons should be interpreted together with the number of evaluated rows and prediction coverage. In particular, ALS may omit predictions for unseen users or items when cold-start handling is enabled.

The scalability experiment compares fixed data samples at 25%, 50%, and 100% of the available explicit-rating records. It repeats each size to report average runtime and standard deviation. Runtime values are environment-dependent and should be regenerated on the machine used for the final report.

## Results

Generated metrics and figures are stored under `analysis/results/`. These are experiment outputs, not a substitute for rerunning the scripts. Regenerate them after changing code, input data, Spark settings, or model parameters.

## Reproducibility checklist

- [ ] Confirm all three raw CSV files are present under `data/raw/`.
- [ ] Confirm Java and PySpark versions.
- [ ] Install dependencies from `requirements-bigdata.txt`.
- [ ] Run preprocessing from the repository root.
- [ ] Run quality checks and EDA.
- [ ] Run each model and save its metrics.
- [ ] Run the repeated scalability experiment.
- [ ] Regenerate visualizations.
- [ ] Record the actual machine, Java, Python, and Spark configuration in the final report.
- [ ] Verify that the report's tables and figures match the latest generated result files.

## Notes

- `requirements.txt`, `Dockerfile`, and `docker-compose.yml` currently belong to the separate Flask/PostgreSQL application setup. Use `requirements-bigdata.txt` for this Spark analytics project.
- Do not interpret historical result files as proof that the current code runs successfully in a clean environment.
- The project report should document the research questions, literature review, data-quality decisions, model limitations, and the interpretation of every experiment.

## Author

Mohammed Muneer Yaseen
