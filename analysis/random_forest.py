import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
import os


def evaluate_feature_importances(X, y, feature_names, name, directory,n_splits=5, random_state=42):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    importances = []
    f1_macro_scores = []
    f1_weighted_scores = []
    fold_reports = []

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test =X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = RandomForestClassifier(random_state=random_state)

        ros = RandomOverSampler(random_state=random_state)
        X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

        clf.fit(X_train_resampled, y_train_resampled)
        y_pred = clf.predict(X_test)

        importances.append(clf.feature_importances_)
        f1_macro_scores.append(f1_score(y_test, y_pred, average='macro'))
        f1_weighted_scores.append(f1_score(y_test, y_pred, average='weighted'))

        fold_reports.append(classification_report(y_test, y_pred, output_dict=True))

    importances = np.array(importances)
    mean_importance = importances.mean(axis=0)
    std_importance = importances.std(axis=0)

    importances_df = pd.DataFrame({
        'Feature': feature_names,
        'Mean Importance': mean_importance,
        'Std Importance': std_importance
    }).sort_values(by='Mean Importance', ascending=False)

    f1_scores = pd.DataFrame({
        'F1 Macro': np.mean(f1_macro_scores),
        'F1 Weighted': np.mean(f1_weighted_scores)
    }, index=[0])

    full_report = pd.DataFrame(fold_reports)

    importances_df.to_csv(directory + '/feature_importance_' + name + '.csv', index=False)
    f1_scores.to_csv(directory + '/f1_scores_' + name + '.csv', index=False)
    full_report.to_csv(directory + '/full_report_' + name + '.csv', index=False)


def main():
    suffix = 'final_rev'

    df_results = pd.read_pickle('results/' + suffix + '/cosmo_results.pkl')
    cosmo_params_df = pd.read_csv('analysis v2/filtered_cosmologies_latex_final.csv', index_col='cosmo')

    X = cosmo_params_df.drop(columns=['Notes'])

    df_results = df_results.sort_index(level='cosmo', key=lambda x: x.str[1:].astype(int))

    variables = ['vrel_tan', 'vrel_rad']

    filt = 'Real'
    significant = [df_results.xs((variable,filt), level=('variable','filter'))['significant'].values for variable in variables]

    y_vtan = (significant[0] == True).astype(int)
    y_vrad = (significant[1] == True).astype(int)

    directory = 'results/' + suffix + '/random_forest_' + filt
    os.makedirs(directory, exist_ok=True)

    evaluate_feature_importances(X.values, y_vtan, X.columns,'vrel_tan', directory)
    evaluate_feature_importances(X.values, y_vrad, X.columns,'vrel_rad', directory)


if __name__ == "__main__":
    main()