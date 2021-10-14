#include <stdio.h>
#include <iostream>
#include <vector>

#include "preprocess.h"
#include "predict.h"

int main(int argc, const char *argv[]) {
    std::string text = u8"Karu oli hoolimatu, lampjalgne ja räpane.";


    int error = InitPreprocessor("Estonian", "/usr/share/espeak-ng-data-1.50/");
    if (error != 0) {
        return 1;
    }
    std::vector<int> tokens = PreProcess(text);

    std::string model_path = "model";
    if (argc == 1) {
        model_path = argv[0];
    }

    InitModel(model_path);
    
    auto predicted = Predict(tokens);
    for (int i=0; i<predicted.size(); i++) {
        printf("%d", predicted[i]);
    }
    printf("Count: %d", predicted.size());
}
