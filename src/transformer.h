#include <stdio.h>
#include <string>
#include <string.h>
#include <codecvt>
#include <locale>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <map>
#include <vector>
#include <math.h>

#include <espeak-ng/speak_lib.h>
#include "cppflow/cppflow.h"

#include "Eigen/Eigen"
#include "Eigen/Dense"
#include "librosa.h"
#include <nlopt.hpp>

#include "utils.h"

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
private:
    TransformerConfig config;
    std::map<wchar_t, int> tokenMap;
    cppflow::model* model;
    Eigen::MatrixXd basis;

public:
    Transformer(TransformerConfig newConfig);
    void Synthesize(std::string text);
private:
    std::wstring phonemize(std::string text);
    std::vector<int> tokenize(std::wstring phons);
    Eigen::MatrixXd runModel(std::vector<int> tokens);
    void recreate(Eigen::MatrixXd mel);

    Eigen::MatrixXd nnlsMat(Eigen::MatrixXd B);
    Eigen::VectorXd nnlsVec(Eigen::VectorXd b);
};

struct OptData {
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
};

double normFunc(const std::vector<double>& xRaw, std::vector<double>& gradRaw, void* f_data);