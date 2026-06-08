# 🏠 Property Price Tier Classification

> A machine learning pipeline that automatically classifies real estate listings into **Budget**, **Mid-Range**, or **Luxury** tiers — helping agents and analysts prioritize and segment properties at scale.

---

## 📋 Project Overview

This project builds a supervised classification model trained on structured real estate listing data. Given property features like size, location, amenities, and floor level, the model predicts which price tier a property belongs to with **97%+ cross-validated accuracy**.

**Use case:** A real estate company can integrate this model into their CRM or listing platform to auto-tag incoming properties, enabling faster triage, targeted marketing, and smarter pricing strategy.

---

## 🎯 Price Tiers

| Tier | Price Range (USD) | Description |
|------|------------------|-------------|
| 🟢 Budget | < $200,000 | Starter homes, small apartments |
| 🔵 Mid-Range | $200,000 – $500,000 | Family homes, modern apartments |
| 🔴 Luxury | > $500,000 | Premium villas, penthouse units |

---

## 📁 Project Structure

```
property-classifier/
│
├── classify.py          # Main ML pipeline (train + evaluate + predict)
├── properties.csv       # Dataset (70 labeled properties)
├── results.png          # Output charts (confusion matrix, feature importance)
└── README.md            # This file
```

---

## 🔧 Features Used

| Feature | Description |
|---|---|
| `area_sqft` | Total property area in square feet |
| `num_bedrooms` | Number of bedrooms |
| `num_bathrooms` | Number of bathrooms |
| `parking_spaces` | Available parking spots |
| `floor_level` | Floor number in building |
| `age_years` | Age of the property |
| `distance_to_center_km` | Distance to city center |
| `neighborhood_score` | Area desirability score (1–10) |
| `amenity_score` | Composite: pool + gym + security |
| `bath_to_bed_ratio` | Engineered: bathrooms ÷ bedrooms |

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/your-username/property-classifier.git
cd property-classifier
```

### 2. Install dependencies
```bash
pip install pandas scikit-learn xgboost matplotlib seaborn
```

### 3. Run the classifier
```bash
python classify.py
```

**Output:**
- Accuracy scores for Random Forest and XGBoost
- 5-fold cross-validation results
- Full classification report (precision, recall, F1)
- `results.png` — confusion matrix + feature importance chart
- Demo predictions on 3 new sample properties

---

## 📊 Results

| Model | Test Accuracy | CV Accuracy (5-fold) |
|-------|-------------|----------------------|
| Random Forest | 100% | — |
| **XGBoost** | **100%** | **97.14% ± 3.50%** |

> Cross-validation score is the more reliable metric — it reflects how the model generalizes across different data splits.

### Feature Importance
The top predictors are `distance_to_center_km`, `neighborhood_score`, and `area_sqft` — which aligns with real-world real estate intuition: location and size drive price tier more than any other factor.

---

## 🔮 Predict New Properties

You can easily add new properties to the demo block in `classify.py`:

```python
new_properties = pd.DataFrame({
    'area_sqft':             [1500],
    'num_bedrooms':          [3],
    'num_bathrooms':         [2],
    'parking_spaces':        [1],
    'floor_level':           [6],
    'age_years':             [7],
    'distance_to_center_km': [4.5],
    'neighborhood_score':    [7],
    'amenity_score':         [1],
    'bath_to_bed_ratio':     [0.67]
})
```

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **Pandas** — data manipulation
- **Scikit-learn** — preprocessing, Random Forest, evaluation
- **XGBoost** — gradient boosting classifier
- **Matplotlib / Seaborn** — visualizations

---

## 📌 Future Improvements

- [ ] Add geospatial features (lat/lng clustering)
- [ ] Expand dataset with real scraped listings
- [ ] Build a Flask/FastAPI inference endpoint
- [ ] Deploy as a web app for agent use

---

## 👤 Author

**Freddy**
Penetration Testing Student | Data & Security Enthusiast
