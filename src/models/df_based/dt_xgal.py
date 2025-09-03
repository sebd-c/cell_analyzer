# imports module
from os.path import join
from argparse import ArgumentParser
from pandas import read_csv
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


def run_dt_model_xgal(input_path: str,
                      output_folder: str
                      ) -> None:
    # read data
    data = read_pickle(input_path)

    # set columns of interest
    feature_cols = ['cyto_grayscale_mean', 'cyto_grayscale_median', 'cyto_grayscale_max', 'cyto_grayscale_min', 'cyto_grayscale_sum',
                    'cyto_grayscale_int_density', 'cyto_red_mean', 'cyto_red_median',
                    'cyto_red_max', 'cyto_red_min', 'cyto_red_sum', 'cyto_red_int_density',
                    'cyto_green_mean', 'cyto_green_median', 'cyto_green_max',
                    'cyto_green_min', 'cyto_green_sum', 'cyto_green_int_density',
                    'cyto_blue_mean', 'cyto_blue_median', 'cyto_blue_max', 'cyto_blue_min',
                    'cyto_blue_sum', 'cyto_blue_int_density']
    # feature_cols = ['nuc_grayscale_mean', 'nuc_grayscale_median',
    #                 'nuc_grayscale_max', 'nuc_grayscale_min', 'nuc_grayscale_sum',
    #                 'nuc_grayscale_int_density', 'nuc_red_mean', 'nuc_red_median',
    #                 'nuc_red_max', 'nuc_red_min', 'nuc_red_sum', 'nuc_red_int_density',
    #                 'nuc_green_mean', 'nuc_green_median', 'nuc_green_max', 'nuc_green_min',
    #                 'nuc_green_sum', 'nuc_green_int_density', 'nuc_blue_mean',
    #                 'nuc_blue_median', 'nuc_blue_max', 'nuc_blue_min', 'nuc_blue_sum',
    #                 'nuc_blue_int_density'
    #                 ]

    # drop the ground truth column
    X = data.drop(['cons_xgal'], axis=1)
    X = X[feature_cols]
    y = data['cons_xgal']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # train model
    clf = DecisionTreeClassifier(criterion='entropy',
                                 random_state=0,
                                 max_depth=4,
                                 class_weight=None)

    # fit the model
    clf.fit(X_train, y_train)

    # make predictions
    y_pred = clf.predict(X_test)

    # getting model perfomance metrics
    # confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, normalize="all").ravel()

    # get the accuracy on the test group
    acc_test = accuracy_score(y_test, y_pred)

    # get precision
    precision = precision_score(y_test, y_pred)

    # get recall
    recall = recall_score(y_test, y_pred)

    # get f1-score
    f1 = f1_score(y_test, y_pred)

    # balanced accuracy
    bal_accuracy_test = balanced_accuracy_score(y_test, y_pred)

    # repeat accuracy for train to check for overfitting
    y_pred_train_gini = clf.predict(X_train)

    acc_train = accuracy_score(y_train, y_pred_train_gini)

    bal_accuracy_train = balanced_accuracy_score(y_train, y_pred_train_gini)

    # make a dict
    metrics_dict = {'accuracy_train': acc_train,
                    'bal_accuracy_train': bal_accuracy_train,
                    'accuracy_test': acc_test,
                    'bal_accuracy_test': bal_accuracy_test,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'tp': tp,
                    'tn': tn,
                    'fp': fp,
                    'fn': fn
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
                  class_names={0: 'Negative', 1: 'Positive'},
                  filled=True,
                  fontsize=12)

    plt.show()

    _ = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
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
    run_dt_model_xgal(input_path=input_path,
                      output_folder=output_folder
                      )


######################################################################
# running main function


if __name__ == '__main__':
    main()

######################################################################
# end of current module
