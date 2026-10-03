"""
Week 3 Homework - Mall Customer Segmentation

This is a SCAFFOLD, not a solution. The plumbing is done (imports, data load,
a cluster-describing helper). Each step is marked TODO with a pointer to the
lecture section showing the same pattern.

homework_week3_solution.ipynb in this folder is the worked version - but try it
here first, the reasoning is what is being assessed.

HOW TO USE THIS FILE
    In VS Code, use homework_week3.ipynb instead - same content, native notebook.
    Every "# %%" line starts a new cell. Ctrl+Enter runs one, Shift+Enter runs it
    and advances. Work top to bottom - later cells need earlier variables.

    Written answers go in the triple-quoted ANSWER blocks. Keep them in the file:
    the file is your homework submission, and the brief says the marks are in the
    reasoning, not the plots.
"""

# %% [markdown]
# # Setup

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, adjusted_rand_score

import warnings
warnings.filterwarnings('ignore', message='.*n_init.*')

pd.set_option('display.width', 170)
pd.set_option('display.max_columns', 30)

import sklearn
print('scikit-learn', sklearn.__version__)

# Use these everywhere so your results reproduce.
RANDOM_STATE = 42
INCOME = 'Annual Income (k$)'
SPEND = 'Spending Score (1-100)'

# %% [markdown]
# Helper: describe a set of cluster labels the way marketing would read them.
# You will use this twice in Part 2 and again in Parts 4 and 5.

# %%
def describe_clusters(X_original, labels, title=''):
    """Min/max income and spending per cluster, plus size."""
    if title:
        print(title)
    rows = []
    for c in sorted(set(labels)):
        m = X_original[labels == c]
        rows.append({'cluster': c,
                     'income min': m[INCOME].min(), 'income max': m[INCOME].max(),
                     'spend min': m[SPEND].min(), 'spend max': m[SPEND].max(),
                     'n': len(m)})
    out = pd.DataFrame(rows).set_index('cluster')
    print(out.to_string())
    return out


# %% [markdown]
# # Part 1: Exploration & Preprocessing
#
# ## 1.1 Load the data
#
# TODO: read the CSV and drop `CustomerID` - it is an identifier, and clustering
# on it would be meaningless.

# %%
URL = ('https://raw.githubusercontent.com/SteffiPeTaffy/machineLearningAZ/master/'
       'Machine%20Learning%20A-Z%20Template%20Folder/Part%204%20-%20Clustering/'
       'Section%2025%20-%20Hierarchical%20Clustering/Mall_Customers.csv')

# TODO: your code here


# %% [markdown]
# ## 1.2 Look at it
#
# TODO: print .head(), .info() and .describe().

# %%
# TODO: your code here


# %% [markdown]
# ## 1.3 The spreads
#
# TODO: print the standard deviation of the two features you will cluster on, and
# compute how many times larger the income spread is than the spending spread.
# You will refer back to this number in Part 2.
#
# Before you compute it, predict what you expect - the lecture's section 4.1 sets
# up a strong expectation. Then check whether this dataset actually behaves that
# way. Look at the UNITS of the income column.

# %%
# TODO: your code here


# %% [markdown]
# ANSWER 1.3
#
# How many times larger is the income spread? Is that what you expected from the
# lecture, and if not, why not?

# %%
"""
ANSWER (1.3):

TODO

"""

# %% [markdown]
# # Part 2: Features & Scaling
#
# ## 2.1 Build X and scale it
#
# TODO: X = the two feature columns; X_scaled = StandardScaler().fit_transform(X).
# Print the mean and sd after scaling to confirm they are ~0 and ~1.

# %%
# TODO: your code here


# %% [markdown]
# ## 2.2 Do it wrong on purpose
#
# TODO: fit KMeans(n_clusters=5, init='k-means++', n_init=10, random_state=RANDOM_STATE)
# TWICE - once on the raw X, once on X_scaled. Describe both with the helper above,
# and compare them.
#
# Hint: adjusted_rand_score(labels_raw, labels_scaled) compares two labellings
# without caring that the cluster NUMBERS differ. 1.0 means the same partition.
# Pattern: lecture section 4.1.

# %%
# TODO: your code here


# %% [markdown]
# ## 2.3 Plot both side by side
#
# TODO: two scatter plots, income vs spending, coloured by each labelling.

# %%
# TODO: your code here


# %% [markdown]
# ANSWER 2.2
#
# In two sentences: what did the unscaled version actually segment on?
#
# Report the ARI. If the two agree, explain WHY given your Part 1.3 number - and
# say whether you would still keep the scaler, with a reason.

# %%
"""
ANSWER (2.2):

TODO

"""

# %% [markdown]
# # Part 3: Choosing K
#
# ## 3.1 The elbow
#
# TODO: for K = 1..10, fit KMeans on the SCALED data and record `.inertia_`.
# Build a table with a percentage-improvement column - the elbow is far easier to
# defend with numbers than by eye - then plot WCSS against K.
# Pattern: lecture section 3.1.

# %%
# TODO: your code here


# %% [markdown]
# ## 3.2 The silhouette
#
# TODO: for K = 2..10 (silhouette is undefined at K=1), compute
# silhouette_score(X_scaled, labels) and plot it. Mark the best K.
# Pattern: lecture section 3.2.

# %%
# TODO: your code here


# %% [markdown]
# ANSWER 3
#
# Do the two methods agree? Which K are you taking forward, and why? Quote the
# improvement percentages and the silhouette scores rather than describing the
# shape of the curve.
#
# If they disagree, say what that disagreement tells you.

# %%
"""
ANSWER (3):

TODO

"""

# %% [markdown]
# # Part 4: Final model and strategy
#
# ## 4.1 Fit K=5 and attach the labels
#
# TODO: fit on the scaled data, add the labels to df as a column.
#
# For plotting you need the centroids in ORIGINAL units. The model's
# `.cluster_centers_` are in scaled space, so invert them:
#     scaler = StandardScaler().fit(X)
#     centroids = scaler.inverse_transform(final.cluster_centers_)
# Nobody can interpret a centroid at -0.83.

# %%
# TODO: your code here


# %% [markdown]
# ## 4.2 The scatter plot
#
# TODO: income on x, spending on y, coloured by cluster, centroids marked,
# axes in original units.

# %%
# TODO: your code here


# %% [markdown]
# ## 4.3 The profile table
#
# TODO: for each cluster - mean Age, mean income, mean spending score, and n.
# Hint: df.groupby('kmeans_cluster').agg(...)

# %%
# TODO: your code here


# %% [markdown]
# ANSWER 4 - the segments
#
# For each of the five clusters, two or three sentences:
#   - Who is in it (age, income, spending pattern)?
#   - What would you actually DO about them (product, channel, focus)?
#   - A name a marketing manager would use. "Cluster 3" is not a segment name.
#
# Worth also saying: is spending predictable from income here? Look at whether all
# four high/low combinations exist. The answer shapes which segments matter.

# %%
"""
ANSWER (4):

TODO

"""

# %% [markdown]
# # Part 5: DBSCAN
#
# ## 5.1 min_samples and the k-distance graph
#
# TODO: pick min_samples from the rule of thumb (~2 x dimensions), then plot the
# sorted distance to the min_samples-th nearest neighbour and find the knee.
# Printing a few percentiles of that distance makes the knee easier to locate.
# Pattern: lecture section 5.2.
#
# Hint:
#     nn = NearestNeighbors(n_neighbors=MIN_SAMPLES).fit(X_scaled)
#     distances, _ = nn.kneighbors(X_scaled)
#     k_dist = np.sort(distances[:, -1])

# %%
# TODO: your code here


# %% [markdown]
# ## 5.2 Sweep eps
#
# TODO: for a range of eps around your knee, tabulate the number of clusters, the
# number of noise points, and the silhouette score. Then pick a value and justify
# it FROM THE TABLE, not from one lucky run.
#
# Two traps: silhouette needs at least 2 clusters, and it should be computed on
# the non-noise points only - otherwise the noise label is treated as a cluster.
# Also note the highest silhouette is usually NOT the right answer here; look at
# how many customers are left unassigned.

# %%
# TODO: your code here


# %% [markdown]
# ## 5.3 Fit, plot, profile
#
# TODO: fit DBSCAN with your chosen eps, add labels to df, plot with noise points
# in black, and build the same profile table as Part 4 (excluding noise).

# %%
# TODO: your code here


# %% [markdown]
# ANSWER 5
#
# Which min_samples and eps did you choose, and why that eps rather than the one
# with the best silhouette?
#
# How do DBSCAN's clusters compare with K-Means' five? If one DBSCAN cluster is much
# larger than any K-Means cluster, work out which K-Means segments it absorbed - and
# say what that tells you about whether those segments are separated by real gaps.

# %%
"""
ANSWER (5):

TODO

"""

# %% [markdown]
# ## 5.4 The noise points
#
# TODO: print the rows DBSCAN labelled -1. Include their K-Means cluster for
# comparison. Sort by income - the pattern is easier to see that way.

# %%
# TODO: your code here


# %% [markdown]
# ANSWER 5.4
#
# Who are they? Is "belongs to no segment" a reasonable description of them?
#
# Think about it twice: once as a statement about the geometry, and once as a
# statement about their commercial value. Those two answers may not agree.

# %%
"""
ANSWER (5.4):

TODO

"""

# %% [markdown]
# # Part 6: The comparison
#
# A short paragraph each:
#
# - Which algorithm was easier to tune, and what exactly did you have to choose
#   for each?
# - Which produced segments that are more interpretable for marketing?
# - DBSCAN can refuse to classify a customer. For THIS task, is that a feature or
#   a problem? Would your answer change if the task were fraud detection?
# - Which would you deploy, and why?

# %%
"""
ANSWER (6):

TODO

"""
