#include "include/cppflow/cppflow.h"
#include <vector>

using namespace std;

//cppflow::model globalModel;

// Model input and output names, get with command
// saved_model_cli show --dir model --all
#define MODEL_INPUT "serving_default_input_1:0"
#define MODEL_OUTPUT "StatefulPartitionedCall:14"
#define MEL_CHANNELS 80

int InitModel(string model_path) {
    //cppflow::model model(model_path);
    return 0;
}

vector<float> Predict(vector<int> tokens) {

    cppflow::model model("/home/egert/Prog/TransformerTTS-Cpp/model");
    auto model_shape = model.get_operations();


    std::vector<int64_t> shape (1);
    shape[0] = tokens.size();

    auto input = cppflow::tensor(tokens, shape);
    input = cppflow::cast(input, TF_INT32, TF_FLOAT);
    input = cppflow::expand_dims(input, 0);

    auto output = model({{MODEL_INPUT, input}}, {MODEL_OUTPUT});
    

    //output = cppflow::cast(output, TF_FLOAT, TF_INT32);
    auto shape2 = output[0].shape();
    auto shape2vec = shape2.get_tensor();
    auto shape2data = shape2.get_data<int64_t>();
    
    auto values = output[0].get_data<float>();

    return values;
}