import os
import sys
import warnings
from textwrap import wrap

warnings.filterwarnings('ignore')

# ----- Imports -----
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)

# Try to import xgboost; fallback if unavailable
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

# Try to import shap for explainability (optional). If not available, script will
# use simpler feature importance methods.
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

# PDF generation
try:
    from fpdf import FPDF
except Exception:
    FPDF = None

# ----- Config -----
ARTIFACT_DIR = 'artifacts'
os.makedirs(ARTIFACT_DIR, exist_ok=True)

REPORT_PDF = os.path.join(ARTIFACT_DIR, 'Churn_Classification_Report.pdf')
REPORT_TITLE = 'Predicting Customer Churn — A CDO-ready Classification Report'

DATA_URL = (
    'https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv'
)

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# ----- Utility functions -----

def short_print(msg):
    print('\n' + '='*6 + ' ' + msg + ' ' + '='*6 + '\n')


def save_fig(fig, name):
    path = os.path.join(ARTIFACT_DIR, name)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    return path


# ----- 1) Data loading and description -----
short_print('Load data')

try:
    df = pd.read_csv(DATA_URL)
except Exception as e:
    print('Failed to download dataset. Please ensure you have internet or place the CSV next to the script.')
    raise e

print('Dataset loaded: rows=%d, cols=%d' % df.shape)

# Preliminary description saved for report
DATA_DESCRIPTION = df.describe(include='all').transpose()
DATA_COLUMNS = df.columns.tolist()

# ----- 2) Define main objective -----
# For the report we will include text. Here we store it so it can be injected into PDF.
MAIN_OBJECTIVE = (
    'Main objective: Build a production-ready classification model to predict customer churn '
    '(binary target = Churn). The analysis focuses on both prediction (identify customers at '
    'high risk of churn for targeted retention) and interpretation (understand primary drivers of churn) '
    'so stakeholders can operationalize interventions. The chosen model balances predictive performance '
    'and explainability for stakeholder trust.'
)

# ----- 3) Quick EDA and cleaning -----
short_print('EDA & cleaning')

# Look for missing values
missing = df.isna().sum()
print('Missing per column (top 10):')
print(missing[missing>0].head(10))

# Observations: TotalCharges sometimes has spaces; convert to numeric
if 'TotalCharges' in df.columns:
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(' ', np.nan))

# Drop customerID (identifier)
if 'customerID' in df.columns:
    df.drop(columns=['customerID'], inplace=True)

# Target
TARGET = 'Churn'
if TARGET not in df.columns:
    raise ValueError('Expected target column "Churn" not found in dataset.')

# Convert target to binary
df[TARGET] = df[TARGET].map({'Yes': 1, 'No': 0})

# Impute missing TotalCharges with median
if df['TotalCharges'].isna().sum() > 0:
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# Separate numeric and categorical
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
if TARGET in numeric_cols:
    numeric_cols.remove(TARGET)
categorical_cols = [c for c in df.columns if c not in numeric_cols + [TARGET]]

print('Numeric columns:', numeric_cols)
print('Categorical columns (sample):', categorical_cols[:8])

# Quick class balance
class_balance = df[TARGET].value_counts(normalize=True)
print('\nTarget distribution:\n', class_balance)

# ----- 4) Feature engineering -----
short_print('Feature engineering')

# Simple encoding: convert binary object columns 'Yes'/'No' to 1/0 where appropriate
binary_map = {'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0}
for c in categorical_cols:
    if set(df[c].dropna().unique()) <= set(binary_map.keys()):
        df[c] = df[c].map(binary_map)

# Recompute categorical columns after mapping
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
if TARGET in numeric_cols:
    numeric_cols.remove(TARGET)
categorical_cols = [c for c in df.columns if c not in numeric_cols + [TARGET]]

# For remaining categoricals, use one-hot encoding (drop first to avoid collinearity)
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

print('After encoding: columns=%d' % df_encoded.shape[1])

# ----- 5) Train/test split and scaling -----
short_print('Train/test split')

X = df_encoded.drop(columns=[TARGET])
y = df_encoded[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)

print('Train size:', X_train.shape, 'Test size:', X_test.shape)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----- 6) Modeling: three classifiers -----
short_print('Modeling')

models = {}
model_scores = {}

# 1) Logistic Regression (baseline, interpretable)
lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_STATE)
models['LogisticRegression'] = lr

# 2) Random Forest
rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=RANDOM_STATE)
models['RandomForest'] = rf

# 3) XGBoost if available else GradientBoosting
if XGBOOST_AVAILABLE:
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_STATE)
    models['XGBoost'] = xgb
else:
    gb = GradientBoostingClassifier(random_state=RANDOM_STATE)
    models['GradientBoosting'] = gb

# Train and cross-validate each model using the same CV folds
skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

for name, model in models.items():
    print('\nTraining and CV for', name)
    try:
        scores = cross_val_score(model, X_train_scaled, y_train, cv=skf, scoring='roc_auc')
        model_scores[name] = {'cv_roc_auc_mean': scores.mean(), 'cv_roc_auc_std': scores.std()}
        print('CV ROC AUC mean: %.4f +/- %.4f' % (scores.mean(), scores.std()))
    except Exception as e:
        print('Cross-val failed for', name, e)

    # Fit on full training data
    model.fit(X_train_scaled, y_train)
    models[name] = model

# ----- 7) Evaluate on test set -----
short_print('Evaluation on test set')

def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan,
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred, digits=4)
    }
    return metrics

results = {}
for name, model in models.items():
    print('\nEvaluating', name)
    metrics = evaluate_model(name, model, X_test_scaled, y_test)
    results[name] = metrics
    print('Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, ROC_AUC: {roc_auc:.4f}'.format(**metrics))

# Save evaluation table
eval_table = pd.DataFrame({
    name: {k: v for k, v in results[name].items() if k in ['accuracy','precision','recall','f1','roc_auc']}
    for name in results
}).T

eval_table.to_csv(os.path.join(ARTIFACT_DIR, 'model_evaluation_summary.csv'))

# ----- 8) Model selection logic -----
short_print('Model selection')

# Choose model with highest ROC AUC on test set; if performance difference small (<0.02), prefer interpretability (LogisticRegression)
best_name = None
best_auc = -1
for name in results:
    auc = results[name]['roc_auc'] if not np.isnan(results[name]['roc_auc']) else 0
    if auc > best_auc:
        best_auc = auc
        best_name = name

print('Best model by ROC_AUC on test set:', best_name, 'ROC_AUC=', best_auc)

# If best model is within 0.02 of LogisticRegression, pick LogisticRegression for explainability
if 'LogisticRegression' in results:
    lr_auc = results['LogisticRegression']['roc_auc']
    if (best_auc - lr_auc) < 0.02:
        recommended = 'LogisticRegression'
    else:
        recommended = best_name
else:
    recommended = best_name

print('Recommended model:', recommended)

# ----- 9) Explainability: feature importance / coefficients -----
short_print('Explainability')

feature_importance = {}

# For logistic regression: coefficients (top positive/negative)
if 'LogisticRegression' in models:
    lr = models['LogisticRegression']
    coefs = pd.Series(lr.coef_[0], index=X.columns).sort_values(ascending=False)
    feature_importance['LogisticRegression'] = coefs
    # Save figure for top coefficients
    top_coefs = pd.concat([coefs.head(10), coefs.tail(10)])
    fig, ax = plt.subplots(figsize=(8,6))
    top_coefs.plot(kind='barh', ax=ax)
    ax.set_title('Logistic Regression top positive (top) and negative (bottom) coefficients')
    figpath = save_fig(fig, 'lr_top_coeffs.png')

# For tree models: feature_importances_
for name in ['RandomForest', 'XGBoost', 'GradientBoosting']:
    if name in models:
        model = models[name]
        if hasattr(model, 'feature_importances_'):
            fi = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
            feature_importance[name] = fi
            # plot top 20
            fig, ax = plt.subplots(figsize=(8,6))
            fi.head(20).plot(kind='barh', ax=ax)
            ax.invert_yaxis()
            ax.set_title(f'{name} top 20 feature importances')
            figpath = save_fig(fig, f'{name}_top20_fi.png')

# Optionally use SHAP (if available) for the recommended model
shap_image = None
if SHAP_AVAILABLE and recommended in models:
    try:
        explainer = shap.Explainer(models[recommended], X_train_scaled)
        shap_values = explainer(X_test_scaled[:200])  # sample to save time
        fig = shap.plots.beeswarm(shap_values, show=False)
        # shap plotting returns matplotlib figure in older versions; to be safe we save via shap's API
        plt.savefig(os.path.join(ARTIFACT_DIR, 'shap_beeswarm.png'), bbox_inches='tight')
        shap_image = os.path.join(ARTIFACT_DIR, 'shap_beeswarm.png')
        plt.close()
    except Exception:
        shap_image = None

# ----- 10) Generate report PDF -----
short_print('Generating PDF report')

# Prepare report text sections

# Section: Data summary (concise)
num_rows, num_cols = df.shape
unique_counts = df.nunique().sort_values(ascending=False).head(10)

DATA_SUMMARY = (
    f'Original dataset: {num_rows} rows and {num_cols} columns. '
    f'Target variable "{TARGET}" distribution: \n{class_balance.to_dict()}.'
)

MODELING_SUMMARY = (
    'Models trained on the same training set and evaluated on a hold-out test set. '
    f'Cross-validated (Stratified {CV_FOLDS}-fold) ROC AUC values on training data were: ' +
    ', '.join([f"{name}: {model_scores[name]['cv_roc_auc_mean']:.3f}±{model_scores[name]['cv_roc_auc_std']:.3f}" for name in model_scores])
)

RECOMMENDATION_TEXT = (
    f'Recommended model: {recommended}. Reason: balanced trade-off between predictive performance and interpretability. '
)

KEY_FINDINGS = []
# 1. Model performance summary
KEY_FINDINGS.append('Model performance (test set):')
for name in results:
    m = results[name]
    KEY_FINDINGS.append(f'- {name}: Accuracy={m["accuracy"]:.3f}, Precision={m["precision"]:.3f}, Recall={m["recall"]:.3f}, F1={m["f1"]:.3f}, ROC_AUC={m["roc_auc"]:.3f}')

# 2. Top drivers from recommended model
KEY_FINDINGS.append('\nTop drivers of churn (from model explainability):')
if recommended in feature_importance:
    top_feats = feature_importance[recommended].head(10)
    for feat, val in top_feats.items():
        KEY_FINDINGS.append(f'- {feat}: importance={val:.4f}')
else:
    # fallback to logistic's top features
    lr_top = feature_importance.get('LogisticRegression')
    if lr_top is not None:
        for feat, val in lr_top.head(10).items():
            KEY_FINDINGS.append(f'- {feat}: coef={val:.4f}')

# 3. Business implication
KEY_FINDINGS.append('\nBusiness implications:')
KEY_FINDINGS.append('- Customers with higher monthly charges, multiple lines, and shorter tenure are typically more likely to churn. Targeted retention campaigns (discounts, loyalty programs) can be prioritized using the model risk scores.')

# 4. Flaws & next steps
FLAWS = [
    '- Data quality: some numeric fields (TotalCharges) contained whitespace and required cleaning; additional billing/ticket history would improve fidelity.',
    '- Labeling / temporal leakage: dataset is cross-sectional. For a production model use time-series validation and ensure no leakage from future features.',
    '- Feature gaps: include interaction features (e.g., tenure x monthly charges), usage metrics, customer support contact history, and contract renewal dates.',
    '- Model robustness: evaluate calibration, fairness across demographic groups, and stability over time.',
]

NEXT_STEPS = [
    '- Collect recent and richer behavioral data (product usage, support tickets, payment history).',
    '- Implement time-based cross-validation and retraining schedule; monitor model drift.',
    '- A/B test different retention interventions prioritized by model risk scores.',
    '- Consider probabilistic calibration and a two-stage system (interpretable model + high-performing ensemble) for production.',
]

# Create simple PDF using fpdf if available; otherwise write a markdown file for user to convert.
if FPDF is None:
    print('FPDF not installed. Writing a markdown report instead at artifacts/Churn_Classification_Report.md')
    md_path = os.path.join(ARTIFACT_DIR, 'Churn_Classification_Report.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f'# {REPORT_TITLE}\n\n')
        f.write('## Main Objective\n')
        f.write(MAIN_OBJECTIVE + '\n\n')
        f.write('## Data Summary\n')
        f.write(DATA_SUMMARY + '\n\n')
        f.write('## Modeling Summary\n')
        f.write(MODELING_SUMMARY + '\n\n')
        f.write('## Recommended Model\n')
        f.write(RECOMMENDATION_TEXT + '\n\n')
        f.write('## Key Findings\n')
        for line in KEY_FINDINGS:
            f.write('- ' + line + '\n')
        f.write('\n## Potential Flaws\n')
        for line in FLAWS:
            f.write('- ' + line + '\n')
        f.write('\n## Next Steps\n')
        for line in NEXT_STEPS:
            f.write('- ' + line + '\n')
    print('Markdown report generated at', md_path)
else:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font('Arial', 'B', 14)
    pdf.multi_cell(0, 8, REPORT_TITLE, align='C')
    pdf.ln(4)

    # Objective
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 6, 'Main Objective', ln=True)
    pdf.set_font('Arial', '', 11)
    for line in wrap(MAIN_OBJECTIVE, 110):
        pdf.multi_cell(0, 6, line)
    pdf.ln(3)

    # Data description
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 6, 'Data Summary', ln=True)
    pdf.set_font('Arial', '', 11)
    for line in wrap(DATA_SUMMARY, 110):
        pdf.multi_cell(0, 6, line)
    pdf.ln(3)

    # Modeling summary
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 6, 'Modeling Summary', ln=True)
    pdf.set_font('Arial', '', 11)
    for line in wrap(MODELING_SUMMARY, 110):
        pdf.multi_cell(0, 6, line)
    pdf.ln(3)

    # Recommended model
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 6, 'Recommended Model', ln=True)
    pdf.set_font('Arial', '', 11)
    for line in wrap(RECOMMENDATION_TEXT, 110):
        pdf.multi_cell(0, 6, line)
    pdf.ln(3)

    # Key findings
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 6, 'Key Findings & Insights', ln=True)
    pdf.set_font('Arial', '', 11)
    for line in KEY_FINDINGS:
        for subline in wrap(line, 110):
            pdf.multi_cell(0, 6, subline)
    pdf.ln(3)

    # Flaws and next steps
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 6, 'Model Flaws & Next Steps', ln=True)
    pdf.set_font('Arial', '', 11)
    pdf.multi_cell(0, 6, 'Flaws:')
    for line in FLAWS:
        for subline in wrap(line, 110):
            pdf.multi_cell(0, 6, '- ' + subline)
    pdf.ln(2)
    pdf.multi_cell(0, 6, 'Next steps:')
    for line in NEXT_STEPS:
        for subline in wrap(line, 110):
            pdf.multi_cell(0, 6, '- ' + subline)

    # Insert a few figures if they exist
    pdf.add_page()
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 6, 'Appendix: Plots & Model Artifacts', ln=True)
    pdf.ln(4)

    # Add evaluation table image by creating a quick table plot
    fig, ax = plt.subplots(figsize=(8,3))
    ax.axis('off')
    table = ax.table(cellText=eval_table.round(3).values, colLabels=eval_table.columns, rowLabels=eval_table.index, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    figpath = save_fig(fig, 'eval_table.png')
    # add image
    pdf.image(figpath, w=180)

    # Add feature importance images if present
    for fname in os.listdir(ARTIFACT_DIR):
        if fname.endswith('.png') and fname not in ['eval_table.png']:
            pdf.add_page()
            pdf.set_font('Arial', 'B', 11)
            pdf.cell(0, 6, fname.replace('_', ' '), ln=True)
            pdf.ln(2)
            pdf.image(os.path.join(ARTIFACT_DIR, fname), w=180)

    pdf.output(REPORT_PDF)
    print('PDF report generated at', REPORT_PDF)

# ----- 11) Save models/artifacts -----
short_print('Save artifacts')

import joblib
for name, model in models.items():
    joblib.dump(model, os.path.join(ARTIFACT_DIR, f'{name}.pkl'))

# Save scaler
joblib.dump(scaler, os.path.join(ARTIFACT_DIR, 'scaler.pkl'))

print('All artifacts saved to', ARTIFACT_DIR)

# ----- Final summary printed to console -----
short_print('Done — Summary for reviewer')
print('Project title:', REPORT_TITLE)
print('\nMain objective:')
print(MAIN_OBJECTIVE)
print('\nRecommended model for production:', recommended)
print('\nWhere to find the full PDF report and artifacts: ', ARTIFACT_DIR)
