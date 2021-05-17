#include "library.h"
#include <stdio.h>
#include <iostream>
#include <Eigen>
#include <vector>
#include <random>
#include <experimental/random>

#ifdef __WIN32__
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif

/*Présentation de la fonction
 * description
 * Prend en parametre ->
 * Retourne ->
 * */
DLLEXPORT int toto() {
    return 1;
}


DLLEXPORT float random(float start, float end){
    std::random_device rand_dev;
    std::mt19937 generator(rand_dev());
    std::uniform_int_distribution<int>  distrubution(start, end);
    return distrubution(generator);
}
DLLEXPORT std::vector<float>  create_linear_model(int input_dim) {
    std::vector<float> array;
    std::uniform_int_distribution<int> distribution(-1.0,1.0);
    for(int i = 0; i <=  input_dim + 1; i+=1){
        array.push_back(random(-1.0,1.0));
    }
    return array;
}

DLLEXPORT float predict_linear_model_regression_unefficient_but_more_readable(std::vector<float> model, std::vector<float> sample_inputs){
    float result = 0.0;
    std::vector<float> sample_inputs_copy;
    for (int i = 0; i < sample_inputs.size(); i+=1){
        sample_inputs_copy[i] = sample_inputs [i];
    }
    sample_inputs_copy.insert(sample_inputs_copy.begin(),  1.0);

    for(int i = 0; i < model.size(); i+=1){
        result += model[i] * sample_inputs_copy[i];
    }
    return result;
}

DLLEXPORT float predict_linear_model_regression(std::vector<float> model, std::vector<float> sample_inputs){
    float result = model[0] * 1.0;
    for(int i = 1; i< model.size(); i+=1){
        result += model[i] * sample_inputs[i - 1];
    }
    return  result;
}

DLLEXPORT float predict_linear_model_classification(std::vector<float> model, std::vector<float> sample_inputs){
    if(predict_linear_model_regression(model, sample_inputs) >= 0){
        return 1.0;
    }
    return -1.0;
}

DLLEXPORT std::vector<float> train_classification_rosenblatt_rule_linear_model(std::vector<float> model,std::vector<float> flattened_dataset_inputs, std::vector<float> flattened_dataset_expected_outputs,float alpha =0.001,int iterations_count = 50 ){
    int input_dim = model.size()-1;
    std::vector<float> Xk;
    int samples_count = flattened_dataset_inputs.size();
    for(int i = 0; i < iterations_count ; i+=1){
        int k = std::experimental::randint(0, samples_count - 1);
        Xk.push_back(flattened_dataset_inputs[k * (k + 1) * input_dim]);
        float Yk = flattened_dataset_expected_outputs[k];
        float gXk = predict_linear_model_classification(model, Xk);
        for(int j =1; i<model.size(); i+=1){
            model[j] = alpha *(Yk-gXk) * Xk[j-1];
        }
    }
    return model;
}

DLLEXPORT std::vector<float> train_regression_pseudo_inverse_linear_model(std::vector<float> model,std::vector<float> flattened_dataset_inputs, std::vector<float> flattened_dataset_expected_outputs){
    int input_dim = model.size() - 1;
    int samples_count = flattened_dataset_inputs.size();
    Eigen::MatrixXd  X,Y;
    

}



------------------------------
def train_regression_pseudo_inverse_linear_model(model: [float],flattened_dataset_inputs:[float],flattened_dataset_expected_outputs: [float]):
input_dim = len(model) - 1
samples_count = len(flattened_dataset_inputs) // input_dim

X = np.array(flattened_dataset_inputs)
Y = np.array(flattened_dataset_expected_outputs)

X = np.reshape(X, (samples_count, input_dim))
ones = np.ones((samples_count, 1))
X = np.hstack((ones, X))

Y = np.reshape(Y, (samples_count, 1))
W = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T, X)), X.T), Y)

for i in range(len(model)):
model[i] = W[i][0]

def destroy_linear_model(model: [float]):
del model

