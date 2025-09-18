/*
 * Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its contributors
 *    may be used to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
 * OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include "aoclda.h"
#include <algorithm>
#include <iostream>
#include <format>
#include <vector>

#ifndef DATA_DIR
#define DATA_DIR "data/df_data"
#endif

// These two functions are from testbench. Replace T with float and add _s version for some calls.
// Don't know why y is passed to each of them because they are not used by the caller.

/* Extract features and responses from 2 CSVfiles, from dabench_utils.hpp */
template <typename T>
int read_features_responses_csv(std::string features_file, std::string response_file,
                                T **X, T **y, da_int &n_samples, da_int &n_features,
                                da_order order = column_major, bool inference = true) {

    da_status status;
    da_datastore store;
    char **headers;
    da_datastore_init(&store);
    status =
        da_read_csv_s(store, features_file.c_str(), X, &n_samples, &n_features, &headers);
    if (status != da_status_success) {
        da_datastore_print_error_message(store);
        return 1;
    }
    if (inference) {
        da_int n_feat_res;
        status = da_read_csv_s(store, response_file.c_str(), y, &n_samples, &n_feat_res,
                             &headers);
        if (status != da_status_success) {
            da_datastore_print_error_message(store);
            return 1;
        }
    }
    if (order == row_major) {
        T *X_copy = new T[n_samples * n_features];
        memcpy(X_copy, *X, n_samples * n_features * sizeof(T));
        da_switch_order_copy_s(column_major, n_samples, n_features, X_copy, n_samples, *X,
                             n_features);
    }
    da_datastore_destroy(&store);
    return 0;
}


/* from drivers/da_decision_forest.cpp */
int init_decforest_handle(da_handle *handle, std::string features_file,
                          std::string response_file, float **X, float **y, da_int **yint,
                          da_order order) {

    int err;
    da_int n_samples, n_features;
    err = read_features_responses_csv(features_file, response_file, X, y,
                                                n_samples, n_features, order);
    if (err != 0)
        return err;

    // Pre-processing
    da_int n_class = (da_int)*std::max_element(*y, *y + n_samples) + 1;
    (*yint) = (da_int *)calloc(n_samples, sizeof(da_int));
    for (int i = 0; i < n_samples; i++)
        (*yint)[i] = (da_int)(*y)[i];

    // Initialize the handle and its options
    da_status status = da_handle_init_s(handle, da_handle_decision_forest);
    if (status != da_status_success) {
        da_handle_print_error_message(*handle);
        return 1;
    }
    status = da_forest_set_training_data_s(*handle, n_samples, n_features, n_class, *X,
                                         n_samples, *yint);
    if (status != da_status_success) {
        da_handle_print_error_message(*handle);
        return 1;
    }
    return 0;
}


int main(int argc, char* argv[]) {

    if (argc <= 1) {
        std::cout << "Need a dataset name.\n" << std::endl;
        exit(1);
    }

    std::string name = argv[1];

    // inputs and options
//    std::string name = std::format("hepmass_150K"); // best 0.85866  ,   each tree has about 15k nodes, 7.7k leaves
//    std::string name = std::format("higgs1m"); // 0.737524           ,   230k, 125k
//    std::string name = std::format("susy"); // best 0.801562         ,   600k, 300k
//    std::string name = std::format("airline-ohe"); // 0.78385        ,   15k, 7k
//    std::string name = std::format("mnist"); // 0.9697               ,   10k, 5k
    std::string features_file = DATA_DIR;
    features_file += "/" + name + "_X.csv";
    std::string response_file = DATA_DIR;
    response_file += "/" + name + "_y.csv";
    std::string test_features_file = DATA_DIR;
    test_features_file += "/" + name + "_Xtest.csv";
    std::string test_response_file = DATA_DIR;
    test_response_file += "/" + name + "_ytest.csv";
//    da_order order = row_major; //column_major; 50%
    da_order order = column_major; // 80%

    std::cout << features_file << "\n";


    // Initialize the decision forest class and set options
    da_handle forest_handle = nullptr;
    float *X = nullptr, *y = nullptr;
    da_int *yint = nullptr;
    int err = init_decforest_handle(&forest_handle, features_file, response_file
                                           , &X, &y, &yint, order);
    int pass = !err;
    pass &= da_options_set_string(forest_handle, "scoring function", "gini")
            == da_status_success;
    pass &= da_options_set_int(forest_handle, "maximum depth", 29)
            == da_status_success;
    pass &= da_options_set_int(forest_handle, "seed", 1)
            == da_status_success;
    pass &= da_options_set_int(forest_handle, "number of trees", 100)
            == da_status_success;
    pass &= da_options_set_int(forest_handle, "node minimum samples", 2)
            == da_status_success;
    pass &= da_options_set_string(forest_handle, "bootstrap", "yes")
            == da_status_success;
    pass &= da_options_set_real_s(forest_handle, "bootstrap samples factor", 1.0)
            == da_status_success;
    pass &= da_options_set_string(forest_handle, "features selection", "sqrt")
            == da_status_success;
    pass &= da_options_set_int(forest_handle, "maximum features", 0)
            == da_status_success;
    pass &= da_options_set_real_s(forest_handle, "feature threshold", 1.0e-06)
            == da_status_success;
    pass &= da_options_set_real_s(forest_handle, "minimum split score", 1.0e-05)
            == da_status_success;
    pass &= da_options_set_int(forest_handle, "block size", 256)
            == da_status_success;
    //  not listed in default options in the benchmark          
    pass &= da_options_set_real_s(forest_handle, "minimum split improvement", 0.0)
            == da_status_success;
    pass &= da_options_set_string(forest_handle, "tree building order", "breadth first") // 0.85866
            == da_status_success;
//    pass &= da_options_set_string(forest_handle, "tree building order", "depth first") // 0.85812
//            == da_status_success;


// missing options from here are
//  "tree building order" "minimum split improvement" "check data" and "storage order"
// These may just be options for benchmarking. They are not listed for DA and enable them crashed.            
//    pass &= da_options_set_real_s(forest_handle, "proportion features", 0.1)
//            == da_status_success;
//    pass &= da_options_set_real_s(forest_handle, "minimum impurity decrease", 0.0)
//            == da_status_success;
//    pass &= da_options_set_string(forest_handle, "histogram", "yes")
//            == da_status_success;
//    pass &= da_options_set_int(forest_handle, "maximum bins", 256)
//            == da_status_success;
            
    if (!pass) {
        std::cout << "Something went wrong setting up the decision tree data and "
                     "optional parameters.\n";
        return 1;
    }


    // train
    da_status status = da_forest_fit_s(forest_handle);
    if (status != da_status_success) {
        std::cout << "Failure while fitting the trees.\n";
        return 1;
    }


    // test
    float *X_test = nullptr, *y_test = nullptr;
    da_int ns_test, nf_test;

    err = read_features_responses_csv(test_features_file,
                                      test_response_file, &X_test, &y_test,
                                      ns_test, nf_test, order);
    if (err != 0)
        return err;




    std::vector<da_int> y_pred(ns_test);
    status = da_forest_predict_s(forest_handle, ns_test, nf_test, X_test,
                                 ns_test, y_pred.data());

    // Why do we need this?
    int *yint_test = (da_int *)calloc(ns_test, sizeof(da_int));
    for (int i = 0; i < ns_test; i++)
        yint_test[i] = (da_int)y_test[i];

    float mean_accuracy;
    status = da_forest_score_s(forest_handle, ns_test, nf_test, X_test,
                               ns_test, yint_test, &mean_accuracy);
    std::cout << "Mean accuracy on the test data: " << mean_accuracy << std::endl;


    if (X)
        free(X);
    if (y)
        free(y);
    if (X_test)
        free(X_test);
    if (y_test)
        free(y_test);
    if (yint)
        free(yint);
    if (yint_test)
        free(yint_test);
    da_handle_destroy(&forest_handle);
    return 0;

}

