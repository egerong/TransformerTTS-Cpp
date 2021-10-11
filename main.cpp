#include <stdio.h>
#include <iostream>
#include "phonemize.h"

int main(int argc, const char *argv[]) {
    std::string text = u8"Karu oli lampjalgne, (hoolimatu) ja räpane. Samas ei olnud metsas temast heasüdamlikumat looma.";
    

    int error = InitPhonemizer("Estonian", "/usr/share/espeak-ng-data-1.50/");
    printf("Init: %d\n", error);
    auto phonemes = Phonemize(text);
    std::cout << phonemes << std::endl;
}
