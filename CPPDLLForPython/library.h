#ifndef CPPDLLFORPYTHON_LIBRARY_H
#define CPPDLLFORPYTHON_LIBRARY_H

float random(float,float);
float*  create_linear_model(int);
float predict_linear_model_regression(float*,int,float*);
float predict_linear_model_classification(float*,int,float*);
float* train_classification_rosenblatt_rule_linear_model(float* ,int ,float* , int ,float* , float , int  );





#endif //CPPDLLFORPYTHON_LIBRARY_H
