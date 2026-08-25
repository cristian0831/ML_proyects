# Machine Learning Projects

A collection of applied machine learning projects and study notes, developed during a Master's in Physics at UNAM. The applied projects cover the core problem types of the ML lifecycle — hypothesis testing, classification, regression, and unsupervised learning — several of them shipped as deployable services (FastAPI, Streamlit) rather than notebooks alone. Coursework from Harvard's CS50 AI and *Hands-on Machine Learning* (Aurélien Géron) rounds out the repo as formative practice.

---

## Featured Projects

The five projects below are the applied, portfolio-grade work in this repo — each answers a real question end to end, from raw data to a decision or a deployed service.

### 🧪 A/B Testing — [Cookie Cats Retention Experiment](A-B_Testing/A-B_Testing-CookieCats.md)
Does moving a mobile game's first progression gate from level 30 to level 40 help or hurt player retention? Full experiment-analysis workflow on ~86,000 players: randomization checks, a two-proportion z-test, and a Bonferroni-corrected segmentation analysis.
- **Result:** Day-7 retention drops significantly when the gate moves to level 40 (z = 3.16, p = 0.0016), an effect concentrated entirely in mid-engagement players.
- **Recommendation:** keep the gate at level 30 — translated into a concrete business call, not just a p-value.
- **Skills:** hypothesis testing, confidence intervals, multiple-comparison correction, statistical vs. practical significance.

### 💳 Classification — [Home Credit Default Risk](Classification/credit_risk/home_credit_risk.md)
Predicts whether a loan applicant will repay, using Home Credit's multi-table Kaggle dataset (application, bureau history, previous loans, monthly balances).
- **Result:** ROC-AUC of 0.77 on the Kaggle competition data.
- Extended into a production-style **[FastAPI scoring service](Classification/credit_risk/FastAPI_service/README.md)**: an XGBoost model (tuned with Optuna) behind a `POST /predict` endpoint, with a shared feature-engineering module so training and serving stay consistent, plus a batch-scoring client.
- **Skills:** multi-table feature engineering, gradient boosting, hyperparameter search, model serving/deployment.

### 🏠 Regression — [California Housing Prices](Regression/housing.ipynb)
End-to-end regression project (following Géron's ML workflow) that predicts median house value from California census data — stratified sampling, a full preprocessing pipeline, and comparison across linear, tree-based, and ensemble regressors with cross-validated model selection.
- **Skills:** feature engineering, `Pipeline`/`ColumnTransformer`, cross-validation, model comparison and error analysis.

### 🎯 Unsupervised Learning — [Client Segmentation](<Unsupervised-Learning/Client Segmentation/README.md>) & [Movie Recommendation](Unsupervised-Learning/Movie_recommendation/README.md)
Two unsupervised projects, each shipped as an interactive app rather than a static notebook:
- **[Client Segmentation](<Unsupervised-Learning/Client Segmentation/README.md>)** — KMeans clustering (k=6, chosen by silhouette score, validated with Adjusted Rand Index ≈ 0.99 across reseeded refits) on customer demographics/spending data, identifying which segments are the clearest marketing targets (e.g. "Aspirational Spenders" — lowest income, 3x the spend-to-income ratio of any other group). Served through a **Streamlit dashboard**.
- **[Movie Recommendation](Unsupervised-Learning/Movie_recommendation/README.md)** — content-based recommender over the TMDB 5000 dataset: genres, cast, and director are vectorized (`CountVectorizer`) into a single feature "soup" per movie, and recommendations are ranked by cosine similarity to a user's stated preferences. Served as a **FastAPI** endpoint with a Pydantic-validated schema and a pytest suite.
- **Skills:** clustering, cluster validation, feature vectorization, similarity search, API design.

---

## Full Project Catalog

### Classification

| Project | Description | Tools |
|---|---|---|
| [Home Credit Risk](Classification/credit_risk/) | Multi-table loan repayment prediction (ROC-AUC 0.77) + deployed FastAPI scoring service | Scikit-learn, XGBoost, Optuna, FastAPI |
| [Marketing Campaign](Classification/marketing_campaign/) | Predicts customer response to a marketing campaign using Random Forest; includes ROI estimation and threshold optimization | Scikit-learn |
| [Titanic Survival](Classification/titanic/) | Kaggle competition: predicts passenger survival from age, sex, class, and other features | Scikit-learn |
| [MNIST](Classification/mnist.ipynb) | Handwritten digit recognition | Scikit-learn |
| [Iris Virginica Classifier](Classification/IrisVirginicaClassifier.ipynb) / [SVM variant](Classification/IrisVirginicaClassifierSVM.ipynb) | Binary flower classification, compared across classifiers and an SVM-specific approach | Scikit-learn |

### Regression

| Project | Description |
|---|---|
| [California Housing Prices](Regression/housing.ipynb) | Predicts housing prices from California census data |
| [Student Performance](Regression/student_performance.ipynb) | Predicts student academic performance from behavioral and demographic features |
| [Life Satisfaction](Regression/life_satisfaction.ipynb) | Regression on the OECD life satisfaction index |

### A/B Testing

| Project | Description |
|---|---|
| [Cookie Cats](A-B_Testing/A-B_Testing-CookieCats.md) | Two-proportion z-test on gate placement and player retention, with segmentation and a Bonferroni-corrected follow-up analysis |

### Unsupervised Learning

| Project | Description | Tools |
|---|---|---|
| [Client Segmentation](<Unsupervised-Learning/Client Segmentation/>) | KMeans customer segmentation with cluster validation, served via a Streamlit dashboard | Scikit-learn, Streamlit |
| [Movie Recommendation](Unsupervised-Learning/Movie_recommendation/) | Content-based recommender (genres/cast/director) over TMDB 5000, served via FastAPI | Scikit-learn, FastAPI, pytest |

---

## Coursework & Study Notes

These are **formative, exercise-driven work** — implementations of algorithms and course assignments used to build fundamentals, not standalone applied projects like the ones above.

### CS50's Introduction to AI with Python (Harvard)

| Project | Description |
|---|---|
| [nim/](nim/) | Trains an AI to play Nim optimally via Q-learning (reinforcement learning) |
| [pagerank/](pagerank/) | Implements PageRank via random-surfer simulation and iterative computation |
| [shopping/](shopping/) | Predicts online shopping purchase intent with a nearest-neighbor classifier |
| [traffic/](traffic/) | CNN traffic-sign classifier; experiments with filter sizes, pooling, hidden layers, and dropout |

### Hands-on Machine Learning (Aurélien Géron) — chapter exercises

| Notebook | Topic |
|---|---|
| [Chapter 4](lec_notes_Aurelin/Chap4_Training_Models.ipynb) | Training Models |
| [Chapter 5](lec_notes_Aurelin/Chap5_Support_Vector_Machines.ipynb) | Support Vector Machines |
| [Chapter 6](lec_notes_Aurelin/Chap6_Decision_trees.ipynb) | Decision Trees |
| [Chapter 7](lec_notes_Aurelin/Chap7_Ensemble_Learning_and_Random_Forests.ipynb) | Ensemble Learning and Random Forests |

### Reference Material

- **[CS229 Lecture Notes](CS229_Lecture_Notes.pdf)** — Stanford Machine Learning course notes by Andrew Ng.

---

## Usage

### Requirements

- Python 3.8+
- Jupyter Notebook or JupyterLab
- Core dependencies:

```bash
pip install scikit-learn tensorflow keras pandas numpy matplotlib seaborn statsmodels
```

- The deployed-service projects have their own dependencies — install from each project's `requirements.txt`:

```bash
# Credit Risk API
pip install xgboost fastapi "uvicorn[standard]" pydantic joblib openpyxl requests

# Client Segmentation dashboard
pip install -r "Unsupervised-Learning/Client Segmentation/requirements.txt"

# Movie Recommendation API
pip install -r Unsupervised-Learning/Movie_recommendation/requirements.txt
```

### Running the notebooks

```bash
git clone https://github.com/cristian0831/Machine-Learning-Projects.git
cd Machine-Learning-Projects
jupyter notebook
```

Open any `.ipynb` file from `Classification/`, `Regression/`, `A-B_Testing/`, or `lec_notes_Aurelin/` to explore interactively.

### Running the deployed services

```bash
# Credit Risk API — from Classification/credit_risk/FastAPI_service/
python train.py                       # train once, produces models/credit_risk_artifact.joblib
uvicorn credit_risk_api:app --reload  # serves POST /predict at :8000/docs

# Client Segmentation dashboard — from "Unsupervised-Learning/Client Segmentation/"
streamlit run app.py                  # serves at :8501

# Movie Recommendation API — from Unsupervised-Learning/Movie_recommendation/
python scripts/build_dataset.py       # one-time: build feature artifacts from the TMDB CSVs
uvicorn app.main:app --reload         # serves POST /recommendations at :8000/docs
```

### Running the CS50 AI scripts

```bash
python nim/play.py                                  # train and play against an AI
python pagerank/pagerank.py pagerank/corpus0        # compute page ranks for a corpus
python shopping/shopping.py shopping/shopping.csv   # predict purchase intent
python traffic/traffic.py <data_directory>          # train a CNN on traffic sign images
```

---

## Contributing

Contributions, suggestions, and improvements are welcome. To contribute:

1. Fork the repository.
2. Create a new branch: `git checkout -b feature/your-feature-name`
3. Commit your changes: `git commit -m 'feat: add your feature'`
4. Push to your fork: `git push origin feature/your-feature-name`
5. Open a Pull Request describing what you changed and why.

Please keep notebooks clean (clear outputs before committing) and include a brief markdown description file alongside any new project.

---

## Tech Stack

- **Python** — primary language
- **Scikit-learn** — classical ML models (SVM, Random Forest, KNN, Logistic Regression, KMeans)
- **XGBoost / Optuna** — gradient boosting and hyperparameter search (Credit Risk)
- **TensorFlow / Keras** — deep learning (CNN for traffic sign recognition)
- **FastAPI / Pydantic / uvicorn** — model-serving APIs (Credit Risk, Movie Recommendation)
- **Streamlit** — interactive dashboards (Client Segmentation)
- **Statsmodels** — hypothesis testing (A/B Testing)
- **Pandas / NumPy** — data manipulation and feature engineering
- **Matplotlib / Seaborn** — data visualization
- **Jupyter Notebooks** — interactive analysis and experimentation

---

## Data

Datasets used across projects are stored in [`Data/datasets/`](Data/datasets/), including California housing, OECD life satisfaction, marketing campaign, and student performance data. Kaggle datasets (Home Credit, Cookie Cats, TMDB 5000) are not redistributed here; download links are provided in each project's markdown description file.
