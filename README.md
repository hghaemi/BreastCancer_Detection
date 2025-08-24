# Breast Cancer Classification 🔬

This project applies multiple **machine learning models** to the Breast Cancer Wisconsin dataset (from `sklearn.datasets`) and compares their performance using **Accuracy, Precision, and Recall**.
The models included are:

* Gaussian Naive Bayes (GNB)
* K-Nearest Neighbors (KNN)
* Decision Tree (DT)
* Random Forest (RF)
* Support Vector Machine (SVM)
* Logistic Regression (LR)
* Artificial Neural Network (ANN with MLPClassifier)

The results are visualized using bar plots for each metric.

---

## 📊 Dataset

The dataset is included in **scikit-learn**:

* **Features**: 30 numeric features computed from a digitized image of a breast mass.
* **Target**: Binary classification (Malignant = 0, Benign = 1).
* **Samples**: 569

Reference: [Breast Cancer Wisconsin Dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-dataset)

---

## ⚙️ Installation

Clone this repository:

```bash
git clone https://github.com/hghaemi/BreastCancer_Detection.git
cd BreastCancer_Detection
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate   # Linux / Mac
.venv\Scripts\activate      # Windows
```

Install requirements:

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

Open and run the `bs_detection.ipynb` notebook in a Jupyter environment (e.g., Jupyter Lab, VS Code with Jupyter extension).

The notebook will:

1. Load the Breast Cancer dataset.
2. Split into train/test sets.
3. Train multiple classifiers.
4. Print performance metrics (Accuracy, Precision, Recall).
5. Generate bar plots to compare models.

---

## 📈 Results

The script outputs bar charts comparing:

* **Training Accuracy**
* **Testing Accuracy**
* **Precision**
* **Recall**

Example (may vary depending on hyperparameters):

| Model | Accuracy (Test) | Precision | Recall |
| ----- | --------------- | --------- | ------ |
| GNB   | 92%             | 91%       | 93%    |
| KNN   | 95%             | 96%       | 94%    |
| ...   | ...             | ...       | ...    |

---

## 📦 Requirements

See [requirements.txt](requirements.txt).

---

## 🧑‍💻 Author

* Hossein ([@hghaemi](https://github.com/hghaemi/))

---

## 📜 License

This project is licensed under the MIT License.
