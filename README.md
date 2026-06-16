# Machine Learning Projects

A collection of machine learning projects and study notes developed during a Master's in Physics at UNAM. Projects span supervised learning, neural networks, reinforcement learning, and statistical testing — drawing from Kaggle competitions, Harvard's CS50 AI course, and *Hands-on Machine Learning with Scikit-Learn, Keras and TensorFlow* by Aurélien Géron.

---

## Overview

This repository is organized into several categories based on the type of problem and source material:

### Classification

Supervised learning projects for binary and multi-class classification problems.

| Project | Description | Tools |
|---|---|---|
| [Iris Virginica Classifier](Classification/IrisVirginicaClassifier.ipynb) | Classifies Iris Virginica flowers using multiple classifiers | Scikit-learn |
| [Iris Virginica Classifier (SVM)](Classification/IrisVirginicaClassifierSVM.ipynb) | SVM-based approach to Iris classification | Scikit-learn |
| [MNIST](Classification/mnist.ipynb) | Handwritten digit recognition on the MNIST dataset | Scikit-learn |
| [Titanic Survival](Classification/titanic/) | Kaggle competition: predicts passenger survival based on age, sex, class, and other features | Scikit-learn |
| [Home Credit Risk](Classification/credit_risk/) | Predicts loan repayment ability using multi-table financial data (Kaggle); achieved ROC-AUC of 0.77 | Scikit-learn |
| [Marketing Campaign](Classification/marketing_campaign/) | Predicts customer response to a marketing campaign using Random Forest; includes ROI estimation and threshold optimization | Scikit-learn |

### Regression

Notebooks for continuous-output prediction problems.

| Project | Description |
|---|---|
| [California Housing Prices](Regression/housing.ipynb) | Predicts housing prices from California census data |
| [Life Satisfaction](Regression/life_satisfaction.ipynb) | Regression on the OECD life satisfaction index |
| [Student Performance](Regression/student_performance.ipynb) | Predicts student academic performance from behavioral and demographic features |

### A/B Testing

| Project | Description |
|---|---|
| [Cookie Cats](A-B_Testing/) | Analyzes player retention in the Cookie Cats mobile game when a gate is moved from level 30 to level 40; compares 1-day and 7-day retention across control and treatment groups |

### CS50 AI Projects

Standalone Python projects from Harvard's *CS50's Introduction to Artificial Intelligence with Python*.

| Project | Description |
|---|---|
| [nim/](nim/) | Trains an AI to play the game of Nim optimally using Q-learning (reinforcement learning) |
| [pagerank/](pagerank/) | Implements the PageRank algorithm via random surfer simulation and iterative computation |
| [shopping/](shopping/) | Predicts online shopping purchase intent from session data using a nearest-neighbor classifier |
| [traffic/](traffic/) | Classifies traffic signs from images using a CNN; experiments with filter sizes, pooling, hidden layers, and dropout |

### Study Notes — Aurélien Géron

Jupyter notebooks following the exercises in *Hands-on Machine Learning with Scikit-Learn, Keras and TensorFlow*.

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
- Install dependencies:

```bash
pip install scikit-learn tensorflow keras pandas numpy matplotlib seaborn
```

### Running the Notebooks

Clone the repository and launch Jupyter:

```bash
git clone https://github.com/cristian0831/Machine-Learning-Projects.git
cd Machine-Learning-Projects
jupyter notebook
```

Open any `.ipynb` file from the `Classification/`, `Regression/`, or `lec_notes_Aurelin/` directories to explore the projects interactively.

### Running the Python Scripts (CS50 AI Projects)

Each CS50 project is a self-contained Python script:

```bash
# Nim — train and play against an AI
python nim/play.py

# PageRank — compute page ranks for a corpus
python pagerank/pagerank.py pagerank/corpus0

# Shopping — predict purchase intent
python shopping/shopping.py shopping/shopping.csv

# Traffic — train a CNN on traffic sign images
python traffic/traffic.py <data_directory>
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
- **Scikit-learn** — classical ML models (SVM, Random Forest, KNN, Logistic Regression)
- **TensorFlow / Keras** — deep learning (CNN for traffic sign recognition)
- **Pandas / NumPy** — data manipulation and feature engineering
- **Matplotlib / Seaborn** — data visualization
- **Jupyter Notebooks** — interactive analysis and experimentation

---

## Data

Datasets used across projects are stored in [`Data/datasets/`](Data/datasets/), including California housing, OECD life satisfaction, marketing campaign, and student performance data. Kaggle datasets are not redistributed here; download links are provided in each project's markdown description file.
