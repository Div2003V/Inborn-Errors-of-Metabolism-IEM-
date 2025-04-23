# 🧬 Inborn Errors of Metabolism (IEM) Detection Using Machine Learning

> A Python-based ML system built to assist in early detection and classification of **Inborn Errors of Metabolism (IEM)** — rare but critical genetic disorders.

---

## 📌 What Are Inborn Errors of Metabolism?

Inborn Errors of Metabolism (IEM) are **rare genetic disorders** caused by **defects in specific enzymes** involved in the body's metabolism. These defects can lead to a **buildup of toxic substances** or **deficiency of essential compounds**, often affecting infants or children at an early age.

### 👶 Symptoms may include:
- Developmental delays
- Lethargy or seizures
- Vomiting or unusual urine odor
- Liver enlargement
- Coma or death if undiagnosed

Early diagnosis is **critical** but often difficult due to:
- Rarity of the conditions
- Similarity in symptoms across disorders
- Delayed symptom onset

---

## 🎯 Project Objective

Build an **ML-based pipeline** to:
- Train on structured biochemical + clinical data
- Accurately predict the type or presence of an IEM
- Serve as a diagnostic aid for researchers or clinicians

---

## 🏗️ Project Structure


---

## ⚙️ How It Works

1. **Data Loading**  
   `iem_data.csv` is loaded and basic cleaning is done.

2. **Preprocessing**  
   Features are extracted, scaled, and split into train-test sets.

3. **Model Training**  
   A Random Forest Classifier is trained on the dataset.

4. **Evaluation**  
   Outputs accuracy, classification report, and a confusion matrix plot.

5. **Prediction Interface**  
   Accepts new patient feature data and returns the predicted IEM class.

---

## 🚀 How to Run

> Make sure Python 3.7+ is installed.

1. **Clone the repo:**
   ```bash
   git clone https://github.com/yourusername/iem-ml-project.git
   cd iem-ml-project
