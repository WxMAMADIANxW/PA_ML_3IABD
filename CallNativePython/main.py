from ctypes import *

path_to_shared_library = "../CPPDLLForPython/cmake-build-debug/CPPDLLForPython.dll"

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    my_lib = cdll.LoadLibrary(path_to_shared_library)


    def create_MLP_model(npl, dsize):
        npltype = POINTER(c_int)
        dsizetype = c_int

        my_lib.create_MLP_model.argtypes = [npltype, dsizetype]
        my_lib.create_MLP_model.restypes = c_void_p

        model = my_lib.create_MLP_model(npl, dsize)
        return model

    def train_classification_stochastic_backprop_mlp_model(model, flattened_dataset_inputs,flattened_expected_outputs,alpha = 0.01,iterations_count = 1000):
        model_type = POINTER(c_void_p)
        flattened_dataset_inputs_type = [c_void_p]
        flattened_expected_outputs_type = [c_void_p]

        my_lib.predict_linear_model_regression.argtypes = [model_type, flattened_dataset_inputs_type, flattened_expected_outputs_type, c_float, c_int]
        my_lib.predict_linear_model_regression.restypes = None

    def train_regression_stochastic_backprop_mlp_model(model, flattened_dataset_inputs,flattened_expected_outputs,alpha = 0.01,iterations_count = 1000):
        model_type = POINTER(c_void_p)
        flattened_dataset_inputs_type = [c_void_p]
        flattened_expected_outputs_type = [c_void_p]

        my_lib.predict_linear_model_regression.argtypes = [model_type, flattened_dataset_inputs_type, flattened_expected_outputs_type, c_float, c_int]
        my_lib.predict_linear_model_regression.restypes = None

    def predict_mlp_model_classification(model, sample_inputs):
        model_type = POINTER(c_void_p)
        sample_inputs_type = [c_void_p]

        my_lib.predict_mlp_classification.argtypes = [model_type, sample_inputs_type]
        my_lib.predict_mlp_classification.restypes = c_float

        predict_model = my_lib.predict_mlp_model_classification(model, sample_inputs)

        return predict_model

    def predict_mlp_model_regression(model, sample_inputs):
        model_type = POINTER(c_void_p)
        sample_inputs_type = [c_void_p]

        my_lib.predict_mlp_model_regression.argtypes = [model_type, sample_inputs_type]
        my_lib.predict_mlp_model_regression.restypes = c_float

        predict_model = my_lib.predict_mlp_model_regression(model, sample_inputs)

        return predict_model

