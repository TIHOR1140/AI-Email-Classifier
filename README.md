# AI Email Classifier

A Python project for classifying emails using machine learning. The repository currently contains the prepared email dataset, project structure, and dependency list. The training pipeline, prediction logic, API, and frontend are planned but are not implemented yet.

## Project Status

This project is currently in development.

- Processed email dataset: available
- Data preparation scripts: scaffolded
- Preprocessing and training pipeline: not implemented yet
- Prediction module: not implemented yet
- FastAPI application: not implemented yet
- Frontend: not implemented yet

## Repository Structure

```text
AI-Email-Classifier/
|-- app/
|   `-- main.py                 # Planned FastAPI application entry point
|-- dataset/
|   |-- processed/
|   |   `-- email_dataset.csv   # Prepared dataset
|   `-- raw/                    # Local raw datasets; excluded from Git
|-- frontend/                   # Planned user interface
|-- models/                     # Generated model files are not committed
|-- notebooks/                  # Planned exploratory notebooks
|-- src/
|   |-- check_labels.py
|   |-- inspect_dataset.py
|   |-- predict.py
|   |-- prepare_dataset.py
|   |-- preprocessing.py
|   `-- train.py
|-- .gitignore
|-- README.md
`-- requirements.txt
```

## Dataset

The processed dataset is stored at `dataset/processed/email_dataset.csv` and contains these columns:

| Column | Description |
| --- | --- |
| `subject` | Email subject text |
| `body` | Email body text |
| `label` | Classification label |

The original raw datasets are not included in the GitHub repository because some files exceed GitHub's 100 MB file limit. They are ignored by `.gitignore` and should be kept locally or stored using Git LFS or external dataset storage.

## Requirements

- Python 3.10 or newer
- pip

## Local Setup

From the project root:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On macOS or Linux, activate the environment with:

```bash
source venv/bin/activate
```

## Running the Project

The application and machine-learning scripts are currently scaffolds, so there is no runnable training or prediction command yet. Once the implementation is added, this section should document commands such as dataset preparation, model training, prediction, and starting the API server.

## Planned Workflow

1. Inspect and validate the raw email datasets.
2. Prepare a consistent processed dataset.
3. Clean and vectorize email subject and body text.
4. Train and evaluate a classification model.
5. Save the trained model locally in `models/`.
6. Expose predictions through the FastAPI application.
7. Add a frontend for submitting email text and displaying the result.

## Contributing

1. Create a feature branch.
2. Make focused changes.
3. Add tests for implemented behavior.
4. Update this README when setup or usage changes.
5. Open a pull request.

## License

No license has been selected for this project yet.
