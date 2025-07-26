# 🌦️ Weather Prediction Model using Random Forest (Seattle Weather Dataset)

This project is a machine learning classifier built using the `RandomForestClassifier` from `scikit-learn` to predict the **weather condition** (e.g., Rainy, Sunny, Snow) based on the **date** and **wind speed**.  
It uses the public `seattle-weather.csv` dataset.

---

## 📊 Dataset

The dataset includes:
- **Date**
- **Precipitation**
- **Temp_min / Temp_max**
- **Wind**
- **Weather** (target variable)

We extract features like:
- `day`
- `month`
- `year`
- `wind`

> 📌 Note: You can easily add more features like weekday, temp_max, precipitation, etc., to improve accuracy.

---

## 🧠 Model

We use:
- `RandomForestClassifier` for classification
- `LabelEncoder` to encode the target variable (`weather`)
- Evaluation using `accuracy_score`

---

## 📈 Accuracy

- Current model accuracy: **66%**
- You can improve this by adding more features or trying other classifiers.

---

## 📦 Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/yourusername/weather-prediction-model.git
cd weather-prediction-model
pip install -r requirements.txt
