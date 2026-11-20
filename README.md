# DNS GUARD – DNS Tunneling & Exfiltration Detector

A machine-learning-based cybersecurity prototype that analyzes domain names and classifies them into three categories: **Benign, DGA, and DNS Tunneling**.

## Project Overview

DNS GUARD uses Python and an XGBoost classifier to identify suspicious domain-name patterns through lexical and structural feature extraction.

The system extracts 24 features from a domain name and displays its prediction through an interactive Streamlit dashboard.

**Note:** This is a domain-name analysis prototype. It does not capture live DNS packets or monitor real-time network traffic.

## Key Features

- Extraction of 24 domain-name features
- Three-class classification: Benign, DGA, and DNS Tunneling
- XGBoost-based detection model
- Configurable tunneling detection threshold
- Interactive Streamlit dashboard
- Detection history and CSV export
- Automated testing with pytest

## Technologies Used

- Python
- XGBoost and Scikit-learn
- Pandas and NumPy
- Matplotlib and Seaborn
- Streamlit
- Joblib
- Pytest

## Dataset

The project uses the DNS Threats Dataset from Stratosphere Laboratory, CTU Prague.

- [Dataset on Zenodo](https://zenodo.org/records/6508640)
- [Stratosphere Laboratory Dataset Directory](https://mcfp.felk.cvut.cz/publicDatasets/DNS-Threats-Dataset/)

Class labels:

| Class | Meaning |
|---|---|
| 0 | Benign |
| 1 | DGA |
| 2 | DNS Tunneling |

Raw and processed datasets are excluded from this repository. Download the dataset files and place them in `data/raw/` before running the training pipeline.

## Model Evaluation

The final model is V5, trained using 24 features.

**Validation accuracy: 85%**

Validation results:

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Benign | 0.81 | 0.81 | 0.81 |
| DGA | 0.88 | 0.88 | 0.88 |
| DNS Tunneling | 0.35 | 0.62 | 0.45 |

**Test accuracy: approximately 44%**

Test results:

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Benign | 0.41 | 1.00 | 0.58 |
| DGA | 1.00 | 0.10 | 0.18 |
| DNS Tunneling | 0.10 | 0.59 | 0.17 |

The difference indicates a significant distribution mismatch between training/validation and test data. The test data was also used repeatedly during development, so it should not be treated as an independent final benchmark.

**This model is experimental and is not production-ready.**

## Installation

Clone the repository:

```bash
git clone https://github.com/gayatree-sahoo/DNS-Tunneling-and-Exfiltration-Detector.git
cd DNS-Tunneling-and-Exfiltration-Detector
```

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Dashboard

```bash
streamlit run dashboard/app.py
```

Open `http://localhost:8501` in your browser.

## Training Pipeline

After downloading the dataset and placing the files in `data/raw/`, run:

```bash
python src/create_validation.py
python src/prepare_dataset.py
python src/train_model_v5.py
```

## Run Tests

```bash
python -m pytest tests/ -v
```

The current test suite contains six automated tests.

## Project Limitations

- Analyzes domain names, not live DNS traffic.
- Does not inspect DNS packets or network-level data transfer.
- Model performance on the test set is weak.
- More representative training data and independent evaluation are needed.
- Detection history is stored only for the active dashboard session.

## Future Improvements

- Improve model generalization.
- Evaluate additional DNS tunneling datasets.
- Integrate DNS packet and traffic analysis.
- Add persistent detection history.
- Improve detection performance through further feature engineering.

## Developer

**Gayatree Sahoo**  
B.Tech – Computer Science and Engineering (Cyber Security)

[GitHub Profile](https://github.com/gayatree-sahoo)
