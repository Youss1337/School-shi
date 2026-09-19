"""
Week 2 Homework - The Credit Default Challenge

This is a SCAFFOLD, not a solution. The plumbing is done (imports, data load,
a metrics helper). Each homework step is marked TODO with a pointer to the
lecture section that shows the same pattern.

If you get stuck or want to check an answer, homework_week2_solution.ipynb in
this folder is the worked version - but try it here first, the whole point is
the reasoning.

HOW TO USE THIS FILE
    In VS Code, use homework_week2.ipynb instead - same content, native notebook.
    Every "# %%" line starts a new cell. Ctrl+Enter runs one, Shift+Enter runs it
    and advances. Work top to bottom - later cells need earlier variables.

    Written answers go in the triple-quoted ANSWER blocks. Keep them in the file:
    the file is your homework submission.
"""

# %% [markdown]
# # Setup

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc,
                             accuracy_score, recall_score, precision_score, f1_score,
                             roc_auc_score, classification_report)

import warnings
warnings.filterwarnings('ignore', message=".*'penalty' was deprecated.*")
warnings.filterwarnings('ignore', message=".*Inconsistent values: penalty.*")
warnings.filterwarnings('ignore', message=".*Workbook contains no default style.*")

pd.set_option('display.width', 160)
pd.set_option('display.max_columns', 30)

import sklearn
print('scikit-learn', sklearn.__version__)

# %% [markdown]
# Scoring helper. Results live in a dict keyed by model name, so re-running a cell
# overwrites that model's row instead of quietly appending a duplicate.

# %%
scores = {}


def score(name, model, X_eval, y_eval):
    """Score a fitted classifier on the Default=1 class."""
    pred = model.predict(X_eval)
    proba = model.predict_proba(X_eval)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_eval, pred).ravel()
    scores[name] = {
        'Model': name,
        'Accuracy': accuracy_score(y_eval, pred),
        'Recall': recall_score(y_eval, pred, zero_division=0),
        'Precision': precision_score(y_eval, pred, zero_division=0),
        'F1': f1_score(y_eval, pred, zero_division=0),
        'ROC-AUC': roc_auc_score(y_eval, proba),
        'FN (funded a defaulter)': fn,
        'FP (refused a good customer)': fp,
    }
    return pd.DataFrame(scores.values()).set_index('Model')


def results_table():
    return pd.DataFrame(scores.values()).set_index('Model')


# %% [markdown]
# # Part 1: EDA & Balance
#
# ## 1.1 Load the data
#
# `index_col=0` matters for the same reason as Week 1: the file carries an unnamed
# first column that is only the row number.
#
# Reading .xlsx needs openpyxl: `conda install openpyxl -y`

# %%
URL = ('https://raw.githubusercontent.com/JWarmenhoven/ISLR-python/'
       'master/Notebooks/Data/Default.xlsx')

df = pd.read_excel(URL, index_col=0)

# TODO: encode 'default' and 'student' from 'Yes'/'No' to 1/0.
# Hint: (df['col'] == 'Yes').astype(int)

print(df.head())
print(df.dtypes)

# %% [markdown]
# ## 1.2 The imbalance
#
# TODO: what percentage of customers default?
# Then build the "dumb" model - predict 0 (no default) for everyone - and score its
# accuracy and recall. Pattern: lecture section 3.7.
# Hint: np.zeros(len(df), dtype=int)

# %%
# TODO: your code here


# %% [markdown]
# ANSWER 1.2
#
# What accuracy does the dumb model reach? What is its recall? What does that tell
# you about using accuracy as your metric for the rest of this assignment?

# %%
"""
ANSWER (1.2):

TODO

"""

# %% [markdown]
# ## 1.3 Balance for defaulters vs non-defaulters
#
# TODO: boxplot 'balance' split by 'default'.
# Hint: plt.boxplot([df.loc[df['default'] == 0, 'balance'],
#                    df.loc[df['default'] == 1, 'balance']],
#                   tick_labels=['No default', 'Default'])
# Worth also doing it for 'income' as a contrast - it behaves very differently.

# %%
# TODO: your code here


# %% [markdown]
# ANSWER 1.3
#
# Do defaulters carry higher balances? Quote the two medians. What does income do
# by comparison?

# %%
"""
ANSWER (1.3):

TODO

"""

# %% [markdown]
# ## 1.4 Check the scales
#
# TODO: print df.std(). How many times larger is the spread of income than balance?
# Keep this number - it is the whole explanation for Part 3.1.

# %%
# TODO: your code here


# %% [markdown]
# ANSWER 1.4

# %%
"""
ANSWER (1.4):

TODO

"""

# %% [markdown]
# # Part 2: Logistic Regression & Interpretability
#
# ## 2.1 Stratified 70/30 split
#
# TODO: build X (student, balance, income) and y (default), then split with
# test_size=0.3, random_state=1, stratify=y.
#
# Why stratify: with a 3% base rate an unlucky split can leave you almost no
# defaulters in the test set, and recall becomes noise.

# %%
# TODO: your code here


# %% [markdown]
# ## 2.2 Logistic regression inside a Pipeline
#
# TODO: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)), fit on
# the TRAINING data. Then score it: score('Logistic Regression', logreg, X_test, y_test)
#
# The scaler must be inside the pipeline. Scaling the full dataset before splitting
# leaks the test set into training.

# %%
# TODO: your code here


# %% [markdown]
# ## 2.3 Coefficients
#
# TODO: print the coefficients with their feature names.
# Hint: pd.Series(logreg.steps[-1][1].coef_[0], index=X.columns)
# np.exp(coefs) converts them to odds multipliers per 1 sd, which is easier to
# talk about in business terms.

# %%
# TODO: your code here


# %% [markdown]
# ANSWER 2.3
#
# * Which creates more risk, Balance or Income?
# * Does being a Student increase or decrease risk according to the model?
# * Why are the coefficients comparable to each other now that you scaled, and why
#   would they NOT have been on the raw data?
#
# Worth checking before you answer the student question: compare the model's sign
# against `df.groupby('student')['default'].mean()`. If they disagree, work out
# why before writing - there is a real effect here, not a bug.

# %%
"""
ANSWER (2.3):

TODO

"""

# %% [markdown]
# ## 2.4 Feature selection with L1
#
# TODO: LogisticRegression(penalty='l1', solver='liblinear', C=0.01) inside a
# pipeline with StandardScaler. Which coefficients went to exactly zero?
#
# Worth sweeping a few values of C (try 1.0, 0.1, 0.01, 0.005) to see when the
# zeros appear. Note C is the INVERSE penalty - small C means strong
# regularization, the opposite direction to alpha in Week 1.

# %%
# TODO: your code here


# %% [markdown]
# ANSWER 2.4
#
# Did any coefficients drop to zero? What does that imply about the true drivers
# of default?

# %%
"""
ANSWER (2.4):

TODO

"""

# %% [markdown]
# # Part 3: The Metric Battle
#
# ## 3.1 KNN, scaled and unscaled
#
# TODO: fit two KNN models with k=9 - one in a pipeline with StandardScaler, one
# on the raw features. Report accuracy, recall and AUC for both.

# %%
# TODO: your code here


# %% [markdown]
# ANSWER 3.1
#
# Explain the gap using the standard deviations from 1.4. Why does scaling matter
# so much more for KNN than it did for logistic regression?
#
# Also worth noting: compare the unscaled model's ACCURACY to the dumb model from 1.2.

# %%
"""
ANSWER (3.1):

TODO

"""

# %% [markdown]
# ## 3.2 Confusion matrices and the metric table
#
# TODO: plot both confusion matrices side by side, then print the comparison table.
# Pattern: lecture sections 3.3 and 3.5.
# Hint: ConfusionMatrixDisplay(cm, display_labels=[...]).plot(cmap='Blues', ax=ax[i])
# The score() helper above already collects Recall/Precision/F1/ROC-AUC - call
# results_table() once both models are scored.

# %%
# TODO: your code here


# %% [markdown]
# ANSWER 3.3 - the trade-off
#
# * Recall: which model is safer (misses fewer defaults)?
# * Precision: which is more efficient (falsely accuses fewer good customers)?
# * F1: which gives the best balance?
#
# If two models tie on a metric, say so rather than manufacturing a winner - and
# then ask which metric actually separates them.

# %%
"""
ANSWER (3.3):

TODO

"""

# %% [markdown]
# ## 3.4 ROC curves
#
# TODO: plot both ROC curves on the same axes with their AUCs in the legend.
# Pattern: lecture section 3.4 / 3.5.

# %%
# TODO: your code here


# %% [markdown]
# ANSWER 3.4
#
# Which curve is closer to the top-left? Look at the left-hand region especially -
# low false positive rate is where a real lender operates.

# %%
"""
ANSWER (3.4):

TODO

"""

# %% [markdown]
# ANSWER 3.5 - The Verdict
#
# * Scenario A: conservative bank, fears losing principal, needs high recall.
# * Scenario B: growth bank, fears rejecting good customers, needs high precision.
# * Final: considering Explainability (Part 2) AND Performance (Part 3), which model
#   for a regulated financial institution?
#
# For the final answer, think about what a bank has to tell a rejected applicant.

# %%
"""
ANSWER (3.5):

TODO

"""

# %% [markdown]
# # Part 4: The Threshold
#
# ## 4.1 / 4.2 Sweep the threshold and cost each outcome
#
# A false negative (funded a defaulter) costs EUR 20,000; a false positive
# (refused a good customer) costs EUR 1,000.
#
# TODO: take logreg.predict_proba(X_test)[:, 1], sweep t over np.arange(0.02, 0.61, 0.02),
# and at each t compute FN, FP and total cost. Pattern: lecture section 5.1.
# Hint: pred = (proba >= t).astype(int)
#       tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()

# %%
# TODO: your code here


# %% [markdown]
# ## 4.3 The cost-minimising threshold
#
# TODO: report the cheapest threshold and its cost, the cost at 0.50, and the
# difference. Plotting cost against threshold makes the point far better than a table.

# %%
# TODO: your code here


# %% [markdown]
# ANSWER 4.3
#
# What is the cost-minimising threshold? How much does the default 0.50 cost you?
# What changed about the MODEL between those two numbers?

# %%
"""
ANSWER (4.3):

TODO

"""

# %% [markdown]
# ## 4.4 KNN's probability granularity
#
# TODO: how many distinct values does knn.predict_proba(X_test)[:, 1] take?
# Hint: np.unique(). Compare with the logistic regression. Then run the same cost
# sweep on the KNN probabilities and look at what the cost column does.

# %%
# TODO: your code here


# %% [markdown]
# ANSWER 4.4
#
# How many distinct probabilities does KNN produce, and why that number? Could you
# implement your optimal threshold with it? Two sentences.

# %%
"""
ANSWER (4.4):

TODO

"""
