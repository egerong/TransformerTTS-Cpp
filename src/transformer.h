#include <stdio.h>
#include <string>
#include <string.h>
#include <codecvt>
#include <locale>
#include <algorithm>
#include <iostream>
#include <map>
#include <vector>
#include <math.h>

#include <espeak-ng/speak_lib.h>
#include "cppflow/cppflow.h"
#include "nanosnap/nanosnap.h"
#include "Eigen/Dense"
#include "Eigen/nnls.h"

struct TransformerConfig {
    bool verbose;
    std::string espeakLang;
    std::string espeakDataPath;
    std::string modelPath;
    bool withStress;
    int sampleRate;
    int nMel;
    int nFFT;
    int fMin;
    int fMax;
};

class Transformer {
public:
    std::string error;

    Transformer(TransformerConfig newConfig);
    std::vector<float> Synthesize(std::string text);
private:
    std::map<wchar_t, int> tokenMap;
    cppflow::model* model;
    TransformerConfig config;

    std::wstring phonemize(std::string text);
    std::vector<int> tokenize(std::wstring phons);
    std::vector<float> runModel(std::vector<int> tokens);
    void recreate(std::vector<float> mel);
};