import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, RobustScaler, StandardScaler
from sklearn.preprocessing import normalize
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from transform_data import *
from sklearn.model_selection import train_test_split
from classification_and_evaluation import *
import warnings

# too ignore warnings (by sklearn mostly)
warnings.filterwarnings('ignore')

features_cols_by_paper = ["F_VCAM1", "N stents/CABG  at FU scan", "F_fifth_Hs-C Reactive Protein ", "F_first_MetabolicSindrome",
                   "F_first_CurrentSmoking", "F_fifth_Creatinine ", \
                   "All segments max stenosis classes scan 2 (0=no stenosis, 1=<30%, 2=30-50%, 3=50-70%, 4=>70%)",
                   "F_first_Dyslipidemia", "F_first_Obesity", \
                   "F_first_Hypertension", "Gender", "F_first_DiabetesMellitus", "F_Plaque_Count ",
                   "F_first_FamilyHistoryCHD", "F_first_PastSmoking", \
                          "F_Nstents", "F_first_statins", "F_first_currentSymptoms", "F_Interleukin6", "F_first_bmiClass",
                          "F_CAD_Score", "F_fifth_Uric acid  ", "F_first_StatinsMgDie", "F_Leptin"]
# for feature selection
FEATURE_NUM = 22

# load data
df = pd.read_excel("Statini_progres.xlsx")
df.replace(to_replace="uninterpretable", value=np.NaN, inplace=True)
print(f"Shape of the dataset = {df.shape}")
X = df[df.columns[1:-1]]
y = df[df.columns[-1]]
# plot output distribution
sb.set(style="darkgrid")
fig = plt.figure()
fig.suptitle('Progress distribution among patients', fontsize=16)
ax = sb.countplot(x="progress", data=df)
plt.show()

known_label_indices = df.index[~df["progress"].isin(("n", np.nan))]
unknown_label_indices = df.index[df["progress"].isin(("n", np.nan))]

# fill missing data
repl_cols = {}
str_cols = set()
for col in X:
    repl_value = None
    if type(X[col][0]) in (
            str, bool, np.bool_):
        repl_value = X[col].mode()[0]
        if type(X[col][0]) == str:
            str_cols.add(col)
    elif isinstance(X[col][0], np.int64):
        repl_value = X[col].median()
    else:
        repl_value = X[col].mean()
    if repl_value is not np.NaN:
        repl_cols[col] = repl_value
    else:
        X = X.drop(col, 1)
X.fillna(repl_cols, inplace=True)

# encode features
label_encoder = LabelEncoder()
for col in str_cols:
    X[col] = encode(label_encoder, X[col])
X.to_excel("cleaned_data.xlsx")

# this can be tested too
# X_to_predict = X.loc[unknown_label_indices, :]
# y_to_predict = y.loc[unknown_label_indices]

X = X.loc[known_label_indices, :]
y = np.array(y.loc[known_label_indices]).astype("int")

# balance the dataset
print(f"Data distribution before resampling = {Counter(y)}")
oversample = SMOTE()
X, y = oversample.fit_resample(X, y)
print(f"Data distribution after resampling = {Counter(y)}")

selection_algorithms = ["mRMR", "select_k_best"]
transformers = {"without scaling": X, "minmax": scale_data(MinMaxScaler(), X),
                "maxabs": scale_data(MaxAbsScaler(), X), "normalize": normalize(X, axis=0, norm="l2"),
                "mRMR(24)": X[features_cols_by_paper]}
"""
1) SVM - C∈{10 −3 ,10 −2 ,...,10 3 } and degree of kernel k∈{1,2} for polynomial, and gamma∈{10 −2 ,10 -1 ,0,10 0 ,10 1 } for RBF kernel.
2) KNN - K∈{1,3,5,7}
"""

classification_algorithms = {
    "Logistic regression": LogisticRegression(solver="saga", class_weight="balanced", max_iter=15000,
                                              random_state=0),
    "Naive Bayes": MultinomialNB(),
    "Decision tree": DecisionTreeClassifier(criterion='entropy',
                                            splitter='best', min_samples_split=5,
                                            min_samples_leaf=10, class_weight="balanced"),
    "Random forest": RandomForestClassifier(criterion="gini", max_depth=15),
    "KNN": KNeighborsClassifier(n_neighbors=5), "SVM": SVC(kernel='rbf', C=1, gamma='scale')}

# try every combination
for scale_type, scaled_data in transformers.items():
    if type(scaled_data) != pd.DataFrame:
        scaled_data = pd.DataFrame(scaled_data, columns=X.columns)

    for selection_algo in selection_algorithms:
        # select top k features - the most useful ones
        print("Selecting best features. This may take some time, please wait!")
        try:

            X_selected = select_features(scaled_data, y, selection_algo, FEATURE_NUM) if scale_type != "mRMR(24)" \
                else scaled_data
            # split dataset into training and test
            X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.25)
            for algorithm, model in classification_algorithms.items():
                y_predicted = apply_algorithm({"x": X_train, "y": y_train}, {"x": X_test},
                                              model)

                calculate_and_plot_metrics(
                    f"{algorithm} with selected features by {selection_algo} and scaling tactics = "
                    f"[{scale_type}]", y_test, y_predicted)

        except Exception as e:
            print(f"Some error occured.Original exception message - {str(e)}")
            continue

