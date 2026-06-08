"""
Property Price Tier Classification
====================================
Classifies real estate properties into: Budget / Mid-Range / Luxury
using XGBoost and Scikit-learn pipeline.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, ConfusionMatrixDisplay)
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("=" * 55)
print("  Property Price Tier Classification")
print("=" * 55)

df = pd.read_csv("properties.csv")
print(f"\n✔ Dataset loaded: {df.shape[0]} properties, {df.shape[1]} features")
print(f"\nClass distribution:\n{df['price_tier'].value_counts()}")

# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────
# Price per sqft (useful but won't leak the label)
df['price_per_sqft'] = df['price_usd'] / df['area_sqft']

# Amenity score (composite feature)
df['amenity_score'] = df['has_pool'] + df['has_gym'] + df['has_security']

# Room ratio
df['bath_to_bed_ratio'] = df['num_bathrooms'] / df['num_bedrooms']

features = [
    'area_sqft', 'num_bedrooms', 'num_bathrooms', 'parking_spaces',
    'floor_level', 'age_years', 'distance_to_center_km',
    'neighborhood_score', 'amenity_score', 'bath_to_bed_ratio'
]

X = df[features]
y = df['price_tier']

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"\nClasses: {list(le.classes_)}")

# ─────────────────────────────────────────────
# 3. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.25, random_state=42, stratify=y_encoded
)
print(f"\nTrain size: {len(X_train)} | Test size: {len(X_test)}")

# ─────────────────────────────────────────────
# 4. TRAIN MODELS
# ─────────────────────────────────────────────
print("\n── Training Models ──")

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_acc = accuracy_score(y_test, rf.predict(X_test))
print(f"  Random Forest Accuracy : {rf_acc:.2%}")

# XGBoost
xgb = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric='mlogloss',
    random_state=42
)
xgb.fit(X_train, y_train)
xgb_preds = xgb.predict(X_test)
xgb_acc = accuracy_score(y_test, xgb_preds)
print(f"  XGBoost Accuracy       : {xgb_acc:.2%}  ← Best model")

# Cross-validation
cv_scores = cross_val_score(xgb, X, y_encoded, cv=5, scoring='accuracy')
print(f"\n  5-Fold CV Mean Accuracy: {cv_scores.mean():.2%} ± {cv_scores.std():.2%}")

# ─────────────────────────────────────────────
# 5. DETAILED REPORT
# ─────────────────────────────────────────────
print("\n── Classification Report (XGBoost) ──")
print(classification_report(y_test, xgb_preds, target_names=le.classes_))

# ─────────────────────────────────────────────
# 6. VISUALISATIONS
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Property Price Tier Classification — Results", fontsize=14, fontweight='bold')

# Plot 1: Confusion Matrix
cm = confusion_matrix(y_test, xgb_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(ax=axes[0], colorbar=False, cmap='Blues')
axes[0].set_title("Confusion Matrix")

# Plot 2: Feature Importance
importances = pd.Series(xgb.feature_importances_, index=features).sort_values()
importances.plot(kind='barh', ax=axes[1], color='steelblue')
axes[1].set_title("Feature Importance (XGBoost)")
axes[1].set_xlabel("Importance Score")

# Plot 3: Class Distribution
df['price_tier'].value_counts().plot(
    kind='bar', ax=axes[2],
    color=['#2ecc71', '#3498db', '#e74c3c'],
    edgecolor='white'
)
axes[2].set_title("Class Distribution")
axes[2].set_xlabel("Price Tier")
axes[2].set_ylabel("Count")
axes[2].tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.savefig("results.png", dpi=150, bbox_inches='tight')
print("\n✔ Chart saved → results.png")

# ─────────────────────────────────────────────
# 7. DEMO PREDICTIONS
# ─────────────────────────────────────────────
print("\n── Demo: Predict new properties ──")

new_properties = pd.DataFrame({
    'area_sqft':              [750,   1700,  4000],
    'num_bedrooms':           [2,     3,     6   ],
    'num_bathrooms':          [1,     2,     5   ],
    'parking_spaces':         [0,     2,     4   ],
    'floor_level':            [3,     8,     22  ],
    'age_years':              [18,    5,     1   ],
    'distance_to_center_km':  [9.5,   4.2,   0.8 ],
    'neighborhood_score':     [4,     7,     10  ],
    'amenity_score':          [0,     1,     3   ],
    'bath_to_bed_ratio':      [0.5,   0.67,  0.83]
})

preds = le.inverse_transform(xgb.predict(new_properties))
for i, pred in enumerate(preds):
    print(f"  Property {i+1}: → {pred.upper()}")

print("\n✔ Done. Model ready for deployment.\n")
