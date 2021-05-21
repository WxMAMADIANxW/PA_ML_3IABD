#include "library.h"
#include <stdio.h>
#include <iostream>
#include <Eigen>
#include <vector>
#include <random>
#include <experimental/random>

#if defined( _WIN32) || defined( __WIN32__) || defined( WIN32 ) || defined( __NT__ )
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif


float random(float start, float end){
    std::random_device rand_dev;
    std::mt19937 generator(rand_dev());
    std::uniform_int_distribution<int>  distrubution(start, end);
    return distrubution(generator);
}
float* toFloatArray(std::vector<float> array){
    float* res = (float*)(malloc(sizeof(float)*(array.size())));
    for(int i = 0; i< array.size(); i+=1){
        res[i] = array[i];
    }
    return res;
}

float*  create_linear_model(int input_dim) {
    float* model = (float*)(malloc(sizeof(float)*(input_dim+1)));
    for(int i = 0; i <=  input_dim + 1; i+=1){
        model[i] =random(-1.0,1.0);
    }
    return model;
}

float predict_linear_model_regression(float* model,int model_size,float* sample_inputs){
    float result = model[0] * 1.0;
    for(int i = 1; i < model_size; i+=1){
        result+= model[i] * sample_inputs[i-1];
    }
    return result;
}

float predict_linear_model_classification(float* model, int model_size, float* sample_inputs){
    if(predict_linear_model_regression(model,model_size,sample_inputs) >=0){
        return 1.0;
    }
    return 1.0;
}

float* train_classification_rosenblatt_rule_linear_model(float* model,int model_size,float* flattened_dataset_inputs, int dateset_input_size,float* flattened_dataset_expected_outputs, float alpha = 0.001, int iterations_count = 50 ){
    int input_dim = model_size -1;
    std::vector<float> Xk;
    for(int i= 0; i < iterations_count; i+=1){
        int k = (int)(random(0,dateset_input_size-1));
        Xk.push_back(flattened_dataset_inputs[k * (k + 1) * input_dim]);
        float Yk = flattened_dataset_expected_outputs[k];
        float* Xktab = toFloatArray(Xk);
        float gXk = predict_linear_model_classification(model, model_size, Xktab);
        for(int j = 1; i < model_size; i+=1){
            model[j] = alpha * (Yk-gXk) * Xk[j-1];
        }
    }
    return  model;
}
