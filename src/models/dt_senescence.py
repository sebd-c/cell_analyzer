# imports module
from os.path import join
from argparse import ArgumentParser
from pandas import read_pickle
from pandas import DataFrame
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from src.utils.aux_funcs import print_execution_parameters
from src.utils.aux_funcs import enter_to_continue


##########################################################################################
# auxiliary functions


def run_dt_model_senescence(input_path: str,
                            output_folder: str
                            ) -> None:
    # read data
    data = read_pickle(input_path)

    # set columns of interest
    feature_cols = ['cyto_area', 'cyto_arbox', 'cyto_radra', 'cyto_asp',
                    'cyto_ecc', 'cyto_rou', 'cii', 'nuc_area', 'nuc_arbox',
                    'nuc_radra', 'nuc_asp', 'nuc_ecc', 'nuc_rou', 'nii',
                    ]

    # drop the ground truth column
    X = data.drop(['label'], axis=1)
    X = X[feature_cols]
    y = data['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.3,
                                                        random_state=42,
                                                        stratify=y)

    # class names
    class_names = ['Normal', 'Quiescent', 'Fully Senescent', 'Senescent-like']

    # train model
    clf = DecisionTreeClassifier(criterion='entropy',
                                 random_state=0,
                                 max_depth=3,
                                 ccp_alpha=0.025,
                                 # min_samples_leaf=5,
                                 class_weight='balanced')

    # fit the model
    clf.fit(X_train, y_train)

    # repeat accuracy for train to check for overfitting
    y_pred_train = clf.predict(X_train)

    # metrics for train set

    # confusion matrix train set
    fig, ax = plt.subplots(figsize=(25, 25))
    ConfusionMatrixDisplay.from_predictions(y_train, y_pred_train, ax=ax, normalize='true')
    ax.xaxis.set_ticklabels(class_names)
    ax.yaxis.set_ticklabels(class_names)
    _ = ax.set_title('Classification Confusion Matrix Train Normalized')

    plt.show()

    fig, ax = plt.subplots(figsize=(25, 25))
    ConfusionMatrixDisplay.from_predictions(y_train, y_pred_train, ax=ax)
    ax.xaxis.set_ticklabels(class_names)
    ax.yaxis.set_ticklabels(class_names)
    _ = ax.set_title('Classification Confusion Matrix Train')

    plt.show()
    # get accuracy
    acc_train = accuracy_score(y_train, y_pred_train)

    # get balanced accuracy
    bal_accuracy_train = balanced_accuracy_score(y_train, y_pred_train)

    # get precisions
    global_precision_train = precision_score(y_train, y_pred_train, average='micro')
    weighted_precision_train = precision_score(y_train, y_pred_train, average='weighted')

    # get recalls
    global_recall_train = recall_score(y_train, y_pred_train, average='micro')
    weighted_recall_train = recall_score(y_train, y_pred_train, average='weighted')

    # get f1-score
    global_f1_train = f1_score(y_train, y_pred_train, average='micro')
    weighted_f1_train = f1_score(y_train, y_pred_train, average='weighted')

    # repeat model prediction and metrics for test set
    y_pred = clf.predict(X_test)

    # confusion matrix
    fig, ax = plt.subplots(figsize=(25, 25))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, normalize='true')
    ax.xaxis.set_ticklabels(class_names)
    ax.yaxis.set_ticklabels(class_names)
    _ = ax.set_title('Classification Confusion Matrix Normalized Test')

    plt.show()

    fig, ax = plt.subplots(figsize=(25, 25))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
    ax.xaxis.set_ticklabels(class_names)
    ax.yaxis.set_ticklabels(class_names)
    _ = ax.set_title('Classification Confusion Matrix Test')

    plt.show()

    # get accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # get balanced accuracy
    bal_accuracy = balanced_accuracy_score(y_test, y_pred)

    # get precisions
    global_precision = precision_score(y_test, y_pred, average='micro')
    weighted_precision = precision_score(y_test, y_pred, average='weighted')

    # get recalls
    global_recall = recall_score(y_test, y_pred, average='micro')
    weighted_recall = recall_score(y_test, y_pred, average='weighted')

    # get f1-score
    global_f1 = f1_score(y_test, y_pred, average='micro')
    weighted_f1 = f1_score(y_test, y_pred, average='weighted')

    # make a dict
    metrics_dict = {'accuracy_train': acc_train,
                    'bal_accuracy_train': bal_accuracy_train,
                    'global_precision_train': global_precision_train,
                    'weighted_precision_train': weighted_precision_train,
                    'global_recall_train': global_recall_train,
                    'weighted_recall_train': weighted_recall_train,
                    'global_f1_score_train': global_f1_train,
                    'weighted_f1_score_train': weighted_f1_train,
                    'accuracy_test': accuracy,
                    'bal_accuracy_test': bal_accuracy,
                    'global_precision_test': global_precision,
                    'weighted_precision_test': weighted_precision,
                    'global_recall_test': global_recall,
                    'weighted_recall_test': weighted_recall,
                    'global_f1_score_test': global_f1,
                    'weighted_f1_score_test': weighted_f1
                    }
    print(metrics_dict)

    # assembling contour df, i.e, making a row
    metrics_df = DataFrame(metrics_dict, index=[0])  # noqa

    # getting feature importance
    feature_importance = DataFrame(clf.feature_importances_,
                                   index=feature_cols).sort_values(0, ascending=False)

    # saving metrics and feature dfs
    # create the path to save
    output_metrics = join(output_folder,
                          'metrics_output.csv')

    output_featimp = join(output_folder,
                          'feat_importance_output.csv')
    # saving df
    metrics_df.to_csv(output_metrics, index=False)

    feature_importance.to_csv(output_featimp, index=False)

    # plottings

    feature_importance.head(10).plot(kind='bar')
    plt.show()

    fig = plt.figure(figsize=(400, 400))
    _ = plot_tree(clf,
                  feature_names=feature_cols,
                  class_names={0: 'Normal', 1: 'Quiescent', 2: 'Fully Senescent', 3: 'Senescent-like'},
                  filled=True,
                  fontsize=12)

    plt.show()


#####################################################################
# argument parsing related functions


def get_args_dict() -> dict:
    """
    Parses the arguments and returns a dictionary of the arguments.
    """
    # defining program description
    description = 'runs a decision tree model'

    # creating a parser instance
    parser = ArgumentParser(description=description)

    # adding arguments to parser

    # input csv path
    parser.add_argument('-i', '--input-path',
                        dest='input_path',
                        required=True,
                        help='defines path to input csv cell informations')

    # output folder
    parser.add_argument('-o', '--output-folder',
                        dest='output_folder',
                        required=True,
                        help='defines path to output files')

    # creating arguments dictionary
    args_dict = vars(parser.parse_args())

    # returning the arguments dictionary
    return args_dict


######################################################################
# defining main function


def main():
    """Runs main code."""
    # getting args dict
    args_dict = get_args_dict()

    # getting input csv path
    input_path = args_dict['input_path']

    # getting images extension
    output_folder = args_dict['output_folder']

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)

    # waiting for user input
    enter_to_continue()

    # running function to preprocess images in a folder
    run_dt_model_senescence(input_path=input_path,
                            output_folder=output_folder
                            )


######################################################################
# running main function


if __name__ == '__main__':
    main()

######################################################################
# end of current module
