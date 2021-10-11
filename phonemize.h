#include <stdio.h>
#include <espeak-ng/speak_lib.h>

int InitPhonemizer(const char *language, const char *data_path);
std::string Phonemize(std::string text);