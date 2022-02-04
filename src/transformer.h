#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <fstream>

#include <string>
#include <string.h>
#include <codecvt>
#include <locale>

#include <map>
#include <vector>
//#include <cmath>
#include <complex>
#include <algorithm>
#include <random>

#include <espeak-ng/speak_lib.h>
#include "cppflow/cppflow.h"
#include "torch/script.h"
#include "audiofile.h"

struct TransformerConfig {
    bool verbose;
    std::string espeakLang;
    std::string espeakDataPath;
    std::string modelPath;
    std::string vocoderPath;
    bool withStress;
    int sampleRate;
    int nFFT;
    int nMel;
    int hopLength;
    int winLength;
    int fMin;
    int fMax;
};

class Transformer {
public:
    std::string error;
private:
    TransformerConfig config;
    std::map<wchar_t, int> tokenMap;
    cppflow::model* model;
    torch::jit::script::Module vocoder;

public:
    Transformer(TransformerConfig newConfig);
    void Synthesize(std::string text);
private:
    std::wstring phonemize(std::string text);
    std::vector<int> tokenize(std::wstring phons);
    std::vector<float> runModel(std::vector<int> tokens);

    std::vector<float> vocode(std::vector<float> mel);

    bool saveWAV(std::string filename, std::vector<float> data);
};

void Test(void);
