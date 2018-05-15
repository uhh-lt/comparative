# File list

This folder contains the final version of all files used in the thesis.

## Experiments

* data.csv - the train/development split of the data
* pre_path_full_paths_original_4.csv - as data.csv, but with LexNet String paths in the original setup
* pre_path_middle_paths_unrestricted_16 - as data.csv, but with LexNet String paths (customized)
* full_paths_original_4.csv - calculated paths (original)
* middle_paths_unrestricted_16.csv - calculated paths (customized)


### Results Three Classes

* 14-22 final_version_no_dups 27.log - log file with all folds
* conf-* - confusion matrices for different max_features
* missclassified_binary_False.csv - all misclassified sentences per feature
* graphic_data - classification results prepared for graphic generation
* errors/3_lex.csv - all errors made by LexNet
* errors/3_se.csv - all errors made by InferSent
* errors/3_shared.csv - all errors made by both both features


### Results Binary
* 15-8 final_version_no_dups 49.log - log with all folds
* conf-* - confusion matrices for different max_features
* missclassified_binary_True.csv - all misclassified sentences per feature
* graphic_data - classification results prepared for graphic generation
* errors/2_classes_errors.csv - all errors made in the binary scenario for LexNet and InferSent

## Hold-Out

* hold-out-data.csv - the test split of the data
