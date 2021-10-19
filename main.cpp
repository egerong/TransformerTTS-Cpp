#include <stdio.h>
#include <iostream>
#include <vector>

#include "transformer.h"

int main(int argc, const char* argv[]) {
    std::string text = u8"Karu oli hoolimatu, lampjalgne ja räpane.";

    Transformer transformer(
        "Estonian",
        "/usr/share/espeak-ng-data-1.50/",
        "model"
    );

    auto predicted = transformer.Synthesize(text);
    for (int i = 0; i < predicted.size(); i++) {
        if (i % 80 == 0) {
            std::cout << std::endl;
        }
        std::cout << predicted[i] << " " << std::ends;
    }

    return 0;

    std::string model_path = "model";
    if (argc == 1) {
        model_path = argv[0];
    }
}
