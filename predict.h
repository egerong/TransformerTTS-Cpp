#include "include/cppflow/cppflow.h"
#include <vector>

using namespace std;

int InitModel(string model_path);
vector<float> Predict(vector<int> tokens);