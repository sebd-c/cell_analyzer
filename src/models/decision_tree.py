# imports module
from os.path import join
from argparse import ArgumentParser
from pandas import read_csv
from pandas import DataFrame
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.tree import DecisionTreeClassifier
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
    data = read_csv(input_path)

    # set columns of interest
    feature_cols = ['grayscale_mean',
                    'grayscale_median',
                    'grayscale_max',
                    'grayscale_min',
                    'grayscale_sum',
                    'grayscale_int_density',
                    'red_mean',
                    'red_median',
                    'red_max',
                    'red_min',
                    'red_sum',
                    'red_int_density',
                    'green_mean',
                    'green_median',
                    'green_max',
                    'green_min',
                    'green_sum',
                    'green_int_density',
                    'blue_mean',
                    'blue_median',
                    'blue_max',
                    'blue_min',
                    'blue_sum',
                    'blue_int_density']

    # drop the ground truth column
    X = data.drop(['xgal_d'], axis=1)
    X = X[feature_cols]
    y = data['xgal_d']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # train model
    clf_gini = DecisionTreeClassifier(criterion='gini', random_state=0)

    # fit the model
    clf_gini.fit(X_train, y_train)

    # make predictions
    y_pred_gini = clf_gini.predict(X_test)

    # getting model perfomance metrics
    # get the accuracy on the test group
    acc_test = accuracy_score(y_test, y_pred_gini)

    # get precision
    precision = precision_score(y_test, y_pred_gini)

    # get recall
    recall = recall_score(y_test, y_pred_gini)

    # get f1-score
    f1 = f1_score(y_test, y_pred_gini)

    # repeat accuracy for train to check for overfitting
    y_pred_train_gini = clf_gini.predict(X_train)

    acc_train = accuracy_score(y_train, y_pred_train_gini)

    # make a dict
    metrics_dict = {'accuracy_train': acc_train,
                    'accuracy_test': acc_test,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1
                    }

    # assembling contour df, i.e, making a row
    metrics_df = DataFrame(metrics_dict, index=[0])  # noqa

    # getting feature importance
    feature_importance = DataFrame(clf_gini.feature_importances_,
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

    fig = plt.figure(figsize=(25, 20))
    _ = plot_tree(clf_gini,
                  feature_names=feature_cols,
                  class_names={0: 'Negative', 1: 'Positive'},
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
    run_dt_model_xgal(input_path=input_path,
                      output_folder=output_folder
                      )


######################################################################
# running main function


if __name__ == '__main__':
    main()

######################################################################
# end of current module
