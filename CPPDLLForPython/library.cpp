#include "library.h"
#include <stdio.h>
#include <Eigen>
#include <vector>
#include <random>
#include <experimental/random>

#if defined( _WIN32) || defined( __WIN32__) || defined( WIN32 ) || defined( __NT__ )
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif


float *toFloatArray(std::vector<float> array) {
    float *res = (float *) (malloc(sizeof(float) * (array.size())));
    for (int i = 0; i < array.size(); i += 1) {
        res[i] = array[i];
    }
    return res;
}

float random(float start, float end) {
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(start, end);
    return distribution(generator);
}

std::vector<float> create_linear_model(int input_dim) {
    std::vector<float> model;
    for (int i = 0; i < input_dim + 1; i += 1) {
        model.push_back(random(-1.0, 1.0));
    }
    return model;
}

float predict_linear_model_regression(std::vector<float> model, std::vector<float> sample_inputs){
    float result = model[0] * 1.0;
    for(int i = 1; i < model.size(); i+=1){
        result+= model[i] * sample_inputs[i-1];
    }
}

float predict_linear_model_classification(std::vector<float> model, std::vector<float> sample_inputs){
    if(predict_linear_model_regression(model,sample_inputs) >=0)
        return 1.0;
    return -1.0;
}

std::vector<float> train_classification_rosenblatt_rule_linear_model(std::vector<float> model, std::vector<float> flattened_dataset_inputs, std::vector<float> flattened_dataset_expected_outputs,float alpha = 0.001, int iterations_count = 50){
    int input_dim = model.size()-1;
    int samples_count = flattened_dataset_inputs.size() / input_dim;
    std::vector<float> Xk;
    for(int i=0; 0 < iterations_count; i+=1){
        int k = (int)(random(0,flattened_dataset_inputs.size()));
        Xk.push_back(flattened_dataset_inputs[k*(k+1)*input_dim]);
        int Yk = flattened_dataset_expected_outputs[k];
        float gXk = predict_linear_model_classification(model,Xk);
        model[0]+= alpha * (Yk - gXk) * 1.0;
        for(int i = 1; i < model.size(); i+=1){
            model[i] += alpha * (Yk - gXk) * Xk[i - 1];
        }
    }
    return model;
}

std::vector<float> train_regression_pseudo_inverse_linear_model(std::vector<float> model, std::vector<float> flattened_dataset_inputs, std::vector<float> flattened_dataset_expected_outputs) {
    int input_dim = model.size() -1;
    int samples_count = flattened_dataset_inputs.size() / input_dim;

    
}

