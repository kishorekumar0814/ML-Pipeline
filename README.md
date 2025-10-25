# ML Pipeline Project

## Project Overview

This project is a Machine Learning pipeline for text sentiment analysis. It includes data ingestion, preprocessing, feature engineering, model training, and model evaluation. The pipeline is managed using **DVC** for reproducibility and version control of data and models.

## Folder Structure

```
ML Pipeline/
├── data/
│   ├── raw/           # Raw data CSVs
│   ├── processed/     # Preprocessed CSVs
│   └── features/      # Feature-engineered CSVs
├── models/            # Trained model files
├── reports/           # Metrics and evaluation reports
├── src/               # Source code
│   ├── data_ingestion.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_building.py
│   └── model_evaluation.py
├── dvc.yaml           # DVC pipeline definition
├── requirements.txt   # Python dependencies
└── README.md
```

## DVC Pipeline Stages

1. **Data Ingestion**: Loads the CSV dataset and splits it into train/test sets.
2. **Data Preprocessing**: Cleans text, removes stop words, lemmatizes, removes URLs, numbers, punctuations.
3. **Feature Engineering**: Converts text into Bag-of-Words features.
4. **Model Building**: Trains a Gradient Boosting Classifier and saves the model.
5. **Model Evaluation**: Evaluates the model and stores metrics in `reports/metrics.json`.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/kishorekumar0814/ML-Pipeline.git
cd ML-Pipeline
```

2. Create and activate a virtual environment:

```bash
python -m venv env
# Windows
env\Scripts\activate
# Mac/Linux
source env/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Install DVC:

```bash
pip install dvc
```

## Usage

Run the DVC pipeline:

```bash
dvc repro
```

This will execute all stages in order: data ingestion, preprocessing, feature engineering, model building, and evaluation.

## Notes

* The `env/`, `data/`, and `models/` folders are ignored by Git. Large files are tracked using DVC.
* To push data and models to a remote storage:

```bash
dvc remote add -d myremote <remote_storage_url>
dvc push
```

* Metrics are stored in `reports/metrics.json`.

## Author

Kishore Kumar S
