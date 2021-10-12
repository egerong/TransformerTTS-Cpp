#include <stdio.h>
#include <iostream>
#include <vector>
#include "preprocess.h"

int main(int argc, const char *argv[]) {
    std::string text = u8"Karu oli hoolimatu, lampjalgne ja räpane.";


    int error = InitPreprocessor("Estonian", "/usr/share/espeak-ng-data-1.50/");
    if (error != 0) {
        return 1;
    }
    std::vector<int> tokens = PreProcess(text);
    
}
