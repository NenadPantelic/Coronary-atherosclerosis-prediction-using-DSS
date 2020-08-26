from sklearn.feature_selection import SelectKBest, chi2
from pymrmr import mRMR
import sklearn_relief as relief


def encode(encoder, data):
    return encoder.fit_transform(data)


def scale_data(scaler, data):
    return scaler.fit_transform(data)


def select_features(X, y, selection_algorithm="mRMR", num_of_features=10):
    selection_algorithm = selection_algorithm.lower()
    assert selection_algorithm.lower() in ("mrmr", "select_k_best", "rrelief"), "Invalid selection algorithm."
    # X_selected_features = X
    print(f"Selecting features with {selection_algorithm} selection algorithm....")
    if selection_algorithm == "mrmr":
        features = mRMR(X, 'MIQ', num_of_features)
        X_selected_features = X[features]
    elif selection_algorithm == "select_k_best":
        X_selected_features = SelectKBest(chi2, k=num_of_features).fit_transform(X, y)
    # raises KeyError - debug required
    else:
        r = relief.Relief(n_features=num_of_features)  # Will run by default on all processors concurrently
        X_selected_features = r.fit_transform(X, y)
    print("Feature selection finished....")
    return X_selected_features


