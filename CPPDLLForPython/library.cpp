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


float random(float start, float end) {
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(start, end);
    return distribution(generator);
}

float *create_linear_model(int input_dim) {
    float *model = (float *) (malloc(sizeof(float) * (input_dim + 1)));
    for (int i = 0; i <= input_dim + 1; i += 1) {
        model[i] = random(-1.0, 1.0);
    }
    return model;
}

float predict_linear_model_regression(float *model, int model_size, float *sample_inputs) {
    float result = model[0] * 1.0;
    for (int i = 1; i < model_size; i += 1) {
        result += model[i] * sample_inputs[i - 1];
    }
    return result;
}

float predict_linear_model_classification(float *model, int model_size, float *sample_inputs) {
    if (predict_linear_model_regression(model, model_size, sample_inputs) >= 0) {
        return 1.0;
    }
    return 1.0;
}

float *train_classification_rosenblatt_rule_linear_model(float *model, int model_size, float *flattened_dataset_inputs,
                                                         int dateset_input_size,
                                                         float *flattened_dataset_expected_outputs, float alpha = 0.001,
                                                         int iterations_count = 50) {
    int input_dim = model_size - 1;
    std::vector<float> Xk;
    for (int i = 0; i < iterations_count; i += 1) {
        int k = (int) (random(0, dateset_input_size - 1));
        Xk.push_back(flattened_dataset_inputs[k * input_dim]);
        float Yk = flattened_dataset_expected_outputs[k]; /*???*/
        float *Xktab = (float *) (malloc(sizeof(float) * (Xk.size())));
        for (int i = 0; i < Xk.size(); i += 1) {
            Xktab[i] = Xk[i];
        }
        float gXk = predict_linear_model_classification(model, model_size, Xktab);
        for (int j = 1; i < model_size; i += 1) {
            model[j] = alpha * (Yk - gXk) * Xk[j - 1];
        }
    }
    return model;
}

float *train_regression_pseudo_inverse_linear_model(float *model, int model_size, float *flattened_dataset_inputs,
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

    /*???*/

    Eigen::MatrixXf W = (((X.transpose() * X).inverse()) * X.transpose()) * Y;
    model = W.data();
    return model;

}


/*----------------------------------
def train_regression_pseudo_inverse_linear_model(model: [float],
flattened_dataset_inputs:[float],
flattened_dataset_expected_outputs: [float]):
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
 -------------------------------------------*/