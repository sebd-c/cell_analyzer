# fix model for df showing in terminal
identities_matrix_df = pd.DataFrame(identities_matrix,
                                            columns=seqs_names_list_ds_01,
                                            index=seqs_names_list_ds_01)
        with pd.option_context('display.max_rows', None, 'display.max_columns',
                               None):  # more options can be specified also
            print(identities_matrix_df)