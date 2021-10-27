#include "transformer.h"

using namespace std;
using namespace Eigen;

#define MODEL_INPUT "serving_default_input_1:0"
#define MODEL_OUTPUT "StatefulPartitionedCall:14"

#define ALL_PHONEMES " !'(),-.:;?abcdefhijklmnopqrstuvwxyzæçðøħŋœǀǁǂǃɐɑɒɓɔɕɖɗɘəɚɛɜɞɟɠɡɢɣɤɥɦɧɨɪɫɬɭɮɯɰɱɲɳɴɵɶɸɹɺɻɽɾʀʁʂʃʄʈʉʊʋʌʍʎʏʐʑʒʔʕʘʙʛʜʝʟʡʢˈˌːˑ˞βθχᵻⱱ"
#define PUNCTUATION "!,-.:;?()"


// For char to wchar conversion
using convert_t = std::codecvt_utf8<wchar_t>;
wstring_convert<convert_t, wchar_t> strconverter;


Transformer::Transformer(TransformerConfig newConfig) {
    config = newConfig;
    // Init eSpeak
    espeak_Initialize(AUDIO_OUTPUT_SYNCHRONOUS, 1, config.espeakDataPath.c_str(), 0);
    int e = espeak_SetVoiceByName(config.espeakLang.c_str());
    if (e != 0) {
        error = "Failed to initialize espeak, check paths and language";
        return;
    }
    // Init token Map
    wstring all_phonemes = strconverter.from_bytes(ALL_PHONEMES);
    tokenMap.insert(pair<wchar_t, int>('/', 0)); // Padding token
    for (int i = 0; i < all_phonemes.length(); i++) {
        tokenMap.insert(pair<wchar_t, int>(all_phonemes[i], i + 1));
    }
    // Load model
    model = new cppflow::model(config.modelPath);
    // Create MEL basis
    basis = librosa::internal::melfilter(
        config.sampleRate,
        config.nFFT,
        config.nMel,
        config.fMin,
        config.fMax
    ).cast<double>();
}

void Transformer::Synthesize(string text) {

    auto phonemes = phonemize(text);
    auto tokens = tokenize(phonemes);
    auto mel = runModel(tokens);
    recreate(mel);
}

wstring Transformer::phonemize(string text) {
    const char* c_text = text.c_str();
    const char** text_ptr = &c_text;
    const char* p;
    string phon;

    int phonememode = 1;
    phonememode |= espeakPHONEMES_IPA; // Use IPA symbols
    //phonememode |= espeakPHONEMES_TIE; // Use the separator (if set) as a tie
    //phonememode |= ' ' << 8; // Use space as a separator

    const char* next_punct;
    char punct;
    while (*text_ptr) {
        next_punct = strpbrk(*text_ptr, PUNCTUATION);
        p = espeak_TextToPhonemes(
            (const void**)text_ptr,
            espeakCHARS_UTF8,
            phonememode
        );
        phon = phon + p;
        if (next_punct) {
            phon = phon + next_punct[0];
        }
        //strcat(all_phonemes, ",");
        //strcat(all_phonemes, phonemes);
    }
    return strconverter.from_bytes(phon);
}

vector<int> Transformer::tokenize(wstring phons) {
    vector<int> tokens;
    int token;
    for (int i = 0; i < phons.length(); i++) {
        token = tokenMap[phons[i]];
        switch (token) {
        case 117: // Exclude stress symbols
            break;
        case 118:
            break;
        default:
            tokens.push_back(token);
        }
    }
    return tokens;
}

MatrixXd Transformer::runModel(vector<int> tokens) {

    std::vector<int64_t> shape(1);
    shape[0] = tokens.size();

    cppflow::tensor input;
    input = cppflow::tensor(tokens, shape);
    input = cppflow::cast(input, TF_INT32, TF_FLOAT);
    input = cppflow::expand_dims(input, 0);

    auto output = (*model)({ {MODEL_INPUT, input} }, { MODEL_OUTPUT });
    auto values = output[0].get_data<float>();
    auto mat = MatrixXf::Map(values.data(), config.nMel, values.size() / config.nMel);


    //output = cppflow::cast(output, TF_FLOAT, TF_INT32);
    auto shape2 = output[0].shape();
    auto shape2vec = shape2.get_tensor();
    auto shape2data = shape2.get_data<int64_t>();

    return mat.cast<double>();
}

void Transformer::recreate(MatrixXd mel) {
    //* Denormalize (in place)
    auto amp = mel.array().exp().matrix();

    matToCSV(amp, "/home/egert/Prog/TTS-CPP/amp.csv");

    //* MEL to STFT

    auto inverse = nnlsMat(amp);

    matToCSV(inverse, "/home/egert/Prog/TTS-CPP/inverse.csv");




    //* Griffin-Lim
    return;
}

MatrixXd Transformer::nnlsMat(MatrixXd B) {
    VectorXd temp;
    MatrixXd inverse(basis.cols(), B.cols());
    for (int i = 0; i < B.cols(); i++) {
        cout << i << "\t";
        //temp = B.col(i);
        //nnlsVec(temp, inverse.col(i));
        temp = nnlsVec(B.col(i));
        inverse.col(i) = temp;
    }
    return inverse;
}

VectorXd Transformer::nnlsVec(VectorXd b) {
    //VectorXd x = basis.colPivHouseholderQr().solve(b);
    VectorXd x = basis.bdcSvd(ComputeThinU | ComputeThinV).solve(b);
    for (int j = 0; j < x.size(); j++) {
        if (x[j] < 0) {
            x[j] = 0;
        }
    }
    vector<double> xRaw;
    xRaw.resize(x.size());
    VectorXd::Map(xRaw.data(), x.size()) = x;


    nlopt::opt opt(nlopt::LD_LBFGS, x.size());
    OptData data{
        .A = basis,
        .b = b
    };
    opt.set_min_objective(normFunc, &data);
    opt.set_lower_bounds(0);
    opt.set_maxeval(15000);
    opt.set_xtol_abs(1e-8);

    double minf;

    try {
        nlopt::result result = opt.optimize(xRaw, minf);
        cout << minf << endl;
    }
    catch (std::exception& e) {
        std::cout << "nlopt failed: " << e.what() << std::endl;
    }
    x = VectorXd::Map(xRaw.data(), xRaw.size());
    return x;
}

//void nnlsObj(x, shape, A, B)

double normFunc(const vector<double>& xRaw, vector<double>& gradRaw, void* f_data) {
    auto data = (OptData*)f_data;
    auto x = VectorXd::Map(xRaw.data(), xRaw.size());

    auto diff = data->A * x - data->b;
    //auto value = diff.squaredNorm();

    if (!gradRaw.empty()) {
        auto grad = VectorXd::Map(gradRaw.data(), gradRaw.size());
        grad = data->A.transpose() * diff;
    }
    return 0.5 * diff.squaredNorm();
}