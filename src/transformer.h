#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <fstream>

#include <string>
#include <string.h>
#include <codecvt>
#include <locale>
#include <time.h>

#include <map>
#include <vector>
//#include <cmath>
#include <complex>
#include <algorithm>
#include <random>

#include <espeak-ng/speak_lib.h>
#include "cppflow/cppflow.h"
#include "torch/torch.h"
#include "torch/script.h"
#include "audiofile.h"

struct TransformerConfig {
    // General
    bool verbose;
    int cudaVisibleDevices;
    // eSpeak
    std::string espeakLang;
    std::string espeakDataPath;
    // Acoustic model
    std::string modelPath;
    int sampleRate;
    int nMel;
    int tfLogLevel;
    // Vocoder
    std::string vocoderPath;
};

class Transformer {
public:
    std::string error;
private:
    TransformerConfig config;
    bool cudaTorch;
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
