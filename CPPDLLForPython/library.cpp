#include "library.h"
#include <cstdio>
#include <Eigen>
#include <vector>
#include <random>
using namespace std;

#if defined( _WIN32) || defined( __WIN32__) || defined( WIN32 ) || defined( __NT__ )
#define DLLEXPORT extern "C" __declspec(dllexport)
#else
#define DLLEXPORT extern "C"
#endif


/*Partie faite par DJALO Mamadian*/

DLLEXPORT float randomPouet(float start, float end) {
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_real_distribution<float> distribution(start, end);
    return distribution(generator);
}

DLLEXPORT float *create_linear_model(int input_dim) {
    float *model = (float *) (malloc(sizeof(float) * (input_dim + 1)));
    for (int i = 0; i <= input_dim + 1; i += 1) {

        model[i] = randomPouet(-1.0, 1.0);
    }
    return model;
}

DLLEXPORT float predict_linear_model_regression(float *model, int model_size, float *sample_inputs) {
    float result = model[0] * 1.0;
    for (int i = 1; i < model_size; i += 1) {
        result += model[i] * sample_inputs[i - 1];
    }
    return result;
}

DLLEXPORT float predict_linear_model_classification(float *model, int model_size, float *sample_inputs) {
    if (predict_linear_model_regression(model, model_size, sample_inputs) >= 0) {
        return 1.0;
    }
    return 1.0;
}

DLLEXPORT float *train_classification_rosenblatt_rule_linear_model(float* model, int model_size, float* flattened_dataset_inputs,
                                                         int dateset_input_size,
                                                         float* flattened_dataset_expected_outputs, float alpha = 0.001,
                                                         int iterations_count = 50) {
    int input_dim = model_size - 1;
    std::vector<float> Xk;
    for (int i = 0; i < iterations_count; i += 1) {
        int k = (int) (randomPouet(0, dateset_input_size - 1));
        Xk.push_back(flattened_dataset_inputs[k * input_dim]);
        float Yk = flattened_dataset_expected_outputs[k*input_dim];
        float* Xktab = Xk.data();
        float gXk = predict_linear_model_classification(model, model_size, Xktab);
        for (int j = 1; i < model_size; i += 1) {
            model[j] = alpha * (Yk - gXk) * Xktab[j - 1];
        }
    }
    return model;
}

DLLEXPORT float* train_regression_pseudo_inverse_linear_model(float *model, int model_size, float *flattened_dataset_inputs,
                                                    int dateset_input_size, float *flattened_dataset_expected_outputs,
                                                    int dataset_output_size) {
    int input_dim = model_size - 1;
    int samples_count = dateset_input_size / input_dim;

    Eigen::MatrixXf X(dateset_input_size, 1);
    Eigen::Vector3f vectorX(flattened_dataset_inputs);
    X << vectorX;

    Eigen::MatrixXf Y(dataset_output_size, 1);
    Eigen::Vector3f vectorY(flattened_dataset_expected_outputs);
    Y << vectorY;

    Eigen::MatrixXf W = (((X.transpose() * X).inverse()) * X.transpose()) * Y;
    model = W.data();
    return model;

}

/* Partie faite par El Walid Ibrahim*/

int l, i, j, it, k;
float sum_result;

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


