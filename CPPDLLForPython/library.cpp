#include "library.h"
#include <stdio.h>
#include <iostream>
#include <Eigen>
#include <random>
using namespace std;

#ifdef __WIN32__
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif

int l, i, j;

class mlp{

public:
    vector<int> d;
    vector<vector<vector<float>>> W;
    vector<vector<float>> X;
    vector<vector<float>> deltas;
public:
    mlp(vector<int> a,
    vector<vector<vector<float>>> b,
    vector<vector<float>> c,
    vector<vector<float>> e){
        this->d = a;
        this->W = b;
        this->X = c;
        this->deltas = e;
    }
};

float random(float start, float end) {
    default_random_engine generator;
    uniform_real_distribution<float> distribution(start, end);
    return distribution(generator);
}

DLLEXPORT float forward_pass(vector<float>sample_inputs, bool is_classification){

    vector<int> d;
    vector<vector<vector<float>>> W;
    vector<vector<float>> X;

    for ( j = 1; j < d[0]+1 ; ++j) {
        X[0][j] = sample_inputs[j -1];

        for(l = 0; l <=d.size(); l++){
            for ( j = 0; j < d[l]+1 ; ++j) {
                float sum_result = 0.0;
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

DLLEXPORT float train_stochastic_gradient_backpropagation(vector<float>flattened_dataset_inputs,
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

};
DLLEXPORT mlp create_MLP_model (vector<int> npl){

    vector<int> d = npl;
    vector<vector<vector<float>>> W;
    vector<vector<float>> X;
    vector<vector<float>> deltas;

    for( l = 0; l <= d.size(); l++){
        W.resize(W.size()+1);
        if(l == 0){
            continue;
        }
        for ( i = 0; i < d[l-1]+1 ; ++i) {
            W[l].resize(W.size()+1);
            for ( j = 0; j < d[l]+1 ; ++j) {
                W[l][i].push_back(random(-1.0,1.0));
            }
        }
    }

    X;
    for(l = 0; l <=d.size(); l++){
        X.resize(X.size()+1);
        for ( j = 0; j < d[l]+1 ; ++j) {
            X[l].push_back((j == 0) ? 1.0 : 0.0);
        }
    }

    deltas;
    for (l = 0; l <= d.size(); l++){
        for ( j = 0; j < d[l]+1 ; ++j) {
            deltas[l].push_back(0.0);
        }
    }

    mlp MLP = *new mlp(d, W, X, deltas);
    return MLP;

}

DLLEXPORT mlp train_classification_stochastic_backprop_mlp_model (mlp MLP, vector<int>flattened_dataset_inputs,
                                                                  vector<int>flattened_expected_outputs,
                                                                  float alpha = 0.01,
                                                                  int iterations_count = 1000){

}

int main() {


}

/*Présentation de la fonction
 * description
 * Prend en parametre ->
 * Retourne ->
    DLLEXPORT int toto() {
    return 1;
}*/