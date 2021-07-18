#include "library.h"
#include <stdio.h>
#include <iostream>
#include <Eigen>
#include <random>
using namespace std;

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
#define DLLEXPORT extern "C" __declspec(dllexport)
#else
#define DLLEXPORT
#endif

int l, i, j, it, k;
float sum_result;

float randomPouet(float start, float end) {
    default_random_engine generator;
    uniform_real_distribution<float> distribution(start, end);
    return distribution(generator);
}

class mlp{

public:
    int *d;
    int dsize;
    vector<vector<vector<float>>> W;
    vector<vector<float>> X;
    vector<vector<float>> deltas;
public:
    mlp(int *a,
        int f,
        vector<vector<vector<float>>> b,
        vector<vector<float>> c,
        vector<vector<float>> e){
        d = a;
        dsize = f;
        W = b;
        X = c;
        deltas = e;
    }
    void forward_pass(vector<float>sample_inputs, bool is_classification){

        vector<int> d;
        vector<vector<vector<float>>> W;
        vector<vector<float>> X;

        for ( j = 1; j < d[0]+1 ; ++j) {
            X[0][j] = sample_inputs[j -1];

            for(l = 0; l <=d.size(); l++){
                for ( j = 0; j < d[l]+1 ; ++j) {
                    sum_result = 0.0;
                    for (i = 0; i < d[l-1]+1; i++){
                        sum_result += W[l][i][j] * X[l-1][i];
                    }
                    X[l][j] = sum_result;
                    if (l < d.size()-1 || l < is_classification ){
                        X[l][j] = tanh(X[l][j]);
                    }
                }
            }
        }
    }

    void train_stochastic_gradient_backpropagation(vector<float>flattened_dataset_inputs,
                                                   vector<float>flattened_expected_outputs,
                                                   bool is_classification,
                                                   float alpha = 0.01,
                                                   int iterations_count = 1000){
        vector<int> d;
        vector<vector<vector<float>>> W;
        vector<vector<float>> X;
        vector<vector<float>> deltas;

        int L = d.size()-1;
        int input_dim = d[0];
        int output_dim = d[L];
        int samples_count = flattened_dataset_inputs.size() / input_dim;

        for(it = 0; it < iterations_count;it ++){
            k = (int) (randomPouet(0, samples_count - 1));
            vector<float> sample_inputs = reinterpret_cast<vector<float, allocator<float>> &&>(flattened_dataset_inputs[k *input_dim,
                    (k + 1) * input_dim]);
            vector<float> sample_expected_outputs = reinterpret_cast<vector<float, allocator<float>> &&>(flattened_expected_outputs[
                    k * output_dim, (k + 1) * output_dim]);

            forward_pass(sample_inputs, is_classification);

            for (j = 1; d[L]+1 ; j++){
                deltas[L][j] = X[L][j] - sample_expected_outputs[j-1];
                if(is_classification){
                    deltas[L][j] = (1 - X[L][j]* X[L][j]) * deltas[L][j];
                }
            }
            for (l = L+1; l>1; l--){
                for (i = 0; i < d[l - 1]+1; i++){
                    sum_result = 0.0;
                    for (j = 1; j< d[l]+1; j++){
                        sum_result += W[l][i][j]* deltas[l][j];
                    }
                    deltas[l-1][i] = (1 - X[l-1][i]* X[l-1][i]) * sum_result;
                }
            }
            for (l = 1; l < L+1; l++){
                for (i = 0; d[l-1]+1; i++){
                    for (j = 1; d[l]+1; j++){
                        W[l][i][j] += -alpha * X[l-1][i]* deltas[l][j];
                    }
                }
            }
        }
    }
};



DLLEXPORT mlp* create_MLP_model (int npl[], int dsize){

    int *d = npl;
    vector<vector<vector<float>>> W;
    vector<vector<float>> X;
    vector<vector<float>> deltas;

    for( l = 0; l <= dsize; l++){
        W.resize(W.size()+1);
        if(l == 0){
            continue;
        }
        for ( i = 0; i < d[l-1]+1 ; ++i) {
            W[l].resize(W.size()+1);
            for ( j = 0; j < d[l]+1 ; ++j) {
                W[l][i].push_back(randomPouet(-1.0,1.0));
            }
        }
    }

    X;
    for(l = 0; l <=dsize; l++){
        X.resize(X.size()+1);
        for ( j = 0; j < d[l]+1 ; ++j) {
            X[l].push_back((j == 0) ? 1.0 : 0.0);
        }
    }

    deltas;
    for (l = 0; l <= dsize; l++){
        for ( j = 0; j < d[l]+1 ; ++j) {
            deltas[l].push_back(0.0);
        }
    }

    mlp* MLP = new mlp(d, dsize, W, X, deltas);
    return MLP;

}

DLLEXPORT void train_classification_stochastic_backprop_mlp_model (mlp* MLP, vector<float>flattened_dataset_inputs,
                                                                  vector<float>flattened_expected_outputs,
                                                                  float alpha = 0.01,
                                                                  int iterations_count = 1000){
    MLP->train_stochastic_gradient_backpropagation(flattened_dataset_inputs, flattened_expected_outputs, true, alpha, iterations_count);
}

DLLEXPORT void train_regression_stochastic_backprop_mlp_model (mlp* MLP, vector<float>flattened_dataset_inputs,
                                                               vector<float>flattened_expected_outputs,
                                                               float alpha = 0.01,
                                                               int iterations_count = 1000){
    MLP->train_stochastic_gradient_backpropagation(flattened_dataset_inputs, flattened_expected_outputs, false, alpha, iterations_count);
}

DLLEXPORT float predict_mlp_model_classification(mlp* MLP, vector<float>sample_inputs){
    MLP->forward_pass(sample_inputs, true);
    return MLP->X[MLP->dsize-1][1];
}

DLLEXPORT float predict_mlp_model_regression(mlp* MLP, vector<float>sample_inputs){
    MLP->forward_pass(sample_inputs, false);
    return MLP->X[MLP->dsize-1][1];
}


/*Présentation de la fonction
 * description
 * Prend en parametre ->
 * Retourne ->
    DLLEXPORT int toto() {
    return 1;
}*/