import argparse
from collections import Counter
from pathlib import Path

from loguru import logger
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier

from coinform_content_analysis.data_loader.data_utils import FEATURES_MAP, get_features, SEMEVAL_TASKA, SEMEVAL_TASKB

EVALUATIONS = Path('evaluations')
TRAINED_MODELS = Path('trained_models')

EVALUATIONS.mkdir(parents=True, exist_ok=True)
TRAINED_MODELS.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

CLASSICAL_MODELS = {
    'Linear SVM': SVC(kernel="linear", C=0.025, random_state=RANDOM_STATE),
    'RBF SVM': SVC(gamma=2, C=1),
    'Linear SVM (squared loss)': LinearSVC(random_state=RANDOM_STATE, C=0.1, dual=True, loss='squared_hinge',
                                           penalty='l2',
                                           tol=0.0001),
    'Logistic Regression': LogisticRegression(random_state=RANDOM_STATE, solver='newton-cg'),
    'Decision Tree': DecisionTreeClassifier(max_depth=7, random_state=RANDOM_STATE, criterion='gini',
                                            min_samples_leaf=2,
                                            min_samples_split=12),
    'Random Forest Classifier': RandomForestClassifier(max_depth=5, n_estimators=10, random_state=RANDOM_STATE)
}


def get_majority_baseline_predictions(y):
    """
    Returns array of len(y) with integer value of majority class
    :param y: gold labels as array of int
    """
    class_counter = Counter(y)
    return [max(class_counter, key=class_counter.get)] * len(y)


def add_baseline(y_task, results, task, collection):
    y_pred = get_majority_baseline_predictions(y_task)
    f1_class = f1_score(y_task, y_pred, average=None)
    results.append({
        'model': 'majority_baseline',
        'feature': None,
        'recall_micro': recall_score(y_task, y_pred, average='micro'),
        'recall_macro': recall_score(y_task, y_pred, average='macro'),
        'precision_macro': precision_score(y_task, y_pred, average='macro'),
        'precision_micro': precision_score(y_task, y_pred, average='micro'),
        'f1_macro': f1_score(y_task, y_pred, average='macro'),
        'f1_micro': f1_score(y_task, y_pred, average='micro'),
        'collection': collection,
        'task': task,
        'f1_class_0': f1_class[0],
        'f1_class_1': f1_class[1],
        'f1_class_2': f1_class[2],
        'f1_class_3': f1_class[3] if len(f1_class) >= 4 else None,
    })


def evaluate_features(file_name='default'):
    y_stance_train = get_features('label', SEMEVAL_TASKA, 'train')
    y_stance_dev = get_features('label', SEMEVAL_TASKA, 'dev')
    y_stance_test = get_features('label', SEMEVAL_TASKA, 'test')

    y_veracity_train = get_features('label', SEMEVAL_TASKB, 'train')
    y_veracity_dev = get_features('label', SEMEVAL_TASKB, 'dev')
    y_veracity_test = get_features('label', SEMEVAL_TASKB, 'test')

    results = []

    add_baseline(y_stance_dev, results, SEMEVAL_TASKA, 'dev')
    add_baseline(y_veracity_dev, results, SEMEVAL_TASKB, 'dev')

    for feature in FEATURES_MAP.keys():
        logger.info('Evaluation for {}'.format(feature))
        X_stance_train = get_features(feature=feature, collection_type='train', task=SEMEVAL_TASKA)
        X_stance_dev = get_features(feature=feature, collection_type='dev', task=SEMEVAL_TASKA)
        X_stance_test = get_features(feature=feature, collection_type='test', task=SEMEVAL_TASKA)

        X_veracity_train = get_features(feature=feature, collection_type='train', task=SEMEVAL_TASKB)
        X_veracity_dev = get_features(feature=feature, collection_type='dev', task=SEMEVAL_TASKB)
        X_veracity_test = get_features(feature=feature, collection_type='test', task=SEMEVAL_TASKB)
        for model_name, model in CLASSICAL_MODELS.items():
            model_path = TRAINED_MODELS / '{task}_{feature}_{model_name}.pkl'.format(task=SEMEVAL_TASKA,
                                                                                     feature=feature,
                                                                                     model_name=model_name)
            eval_feat_helper(X_stance_dev, X_stance_train, X_stance_test, feature, model, model_path, model_name,
                             results,
                             SEMEVAL_TASKA, y_stance_dev,
                             y_stance_train, y_stance_test)

            model_path = TRAINED_MODELS / '{task}_{feature}_{model_name}.pkl'.format(task=SEMEVAL_TASKB,
                                                                                     feature=feature,
                                                                                     model_name=model_name)
            eval_feat_helper(X_veracity_dev, X_veracity_train, X_veracity_test, feature, model, model_path, model_name,
                             results, SEMEVAL_TASKB, y_veracity_dev,
                             y_veracity_train, y_veracity_test)

    model_name = '{}.csv'.format(file_name)
    path = EVALUATIONS / model_name
    results_df = DataFrame(results)
    results_df.to_csv(path)


def eval_feat_helper(X_dev, X_train, X_test, feature, model, model_path, name, results, task, y_stance_dev,
                     y_stance_train, y_stance_test):
    if not model_path.exists():
        model.fit(X_train, y_stance_train)
        joblib.dump(model, model_path)
    else:
        model_obj = joblib.load(model_path)
        y_preds = model_obj.predict(X_dev)
        f1_class = f1_score(y_stance_dev, y_preds, average=None)

        results.append({
            'model': name,
            'feature': feature,
            'recall_micro': recall_score(y_stance_dev, y_preds, average='micro'),
            'recall_macro': recall_score(y_stance_dev, y_preds, average='macro'),
            'precision_macro': precision_score(y_stance_dev, y_preds, average='macro'),
            'precision_micro': precision_score(y_stance_dev, y_preds, average='micro'),
            'f1_macro': f1_score(y_stance_dev, y_preds, average='macro'),
            'f1_micro': f1_score(y_stance_dev, y_preds, average='micro'),
            'collection': 'dev',
            'task': task,
            'f1_class_0': f1_class[0],
            'f1_class_1': f1_class[1],
            'f1_class_2': f1_class[2],
            'f1_class_3': f1_class[3] if len(f1_class) >= 4 else None,
        })
        y_preds = model_obj.predict(X_test)
        eval_func(feature, name, results, task, y_preds, y_stance_test)


def eval_func(feature, name, results, task, y_preds, y_stance_test):
    f1_class = f1_score(y_stance_test, y_preds, average=None)
    results.append({
        'model': name,
        'feature': feature,
        'recall_micro': recall_score(y_stance_test, y_preds, average='micro'),
        'recall_macro': recall_score(y_stance_test, y_preds, average='macro'),
        'precision_macro': precision_score(y_stance_test, y_preds, average='macro'),
        'precision_micro': precision_score(y_stance_test, y_preds, average='micro'),
        'f1_macro': f1_score(y_stance_test, y_preds, average='macro'),
        'f1_micro': f1_score(y_stance_test, y_preds, average='micro'),
        'collection': 'test',
        'task': task,
        'f1_class_0': f1_class[0],
        'f1_class_1': f1_class[1],
        'f1_class_2': f1_class[2],
        'f1_class_3': f1_class[3] if len(f1_class) >= 4 else None,
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate rumoureval models!!!')
    parser.add_argument('--eval', default='features')
    parser.add_argument('--name', default='default')

    args = parser.parse_args()
    if args.eval == 'features':
        evaluate_features(args.name)
