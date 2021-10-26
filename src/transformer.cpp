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
}

vector<float> Transformer::Synthesize(string text) {

    auto phonemes = phonemize(text);
    auto tokens = tokenize(phonemes);
    auto mel = runModel(tokens);
    recreate(mel);
    return mel;
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

vector<float> Transformer::runModel(vector<int> tokens) {

    std::vector<int64_t> shape(1);
    shape[0] = tokens.size();

    cppflow::tensor input;
    input = cppflow::tensor(tokens, shape);
    input = cppflow::cast(input, TF_INT32, TF_FLOAT);
    input = cppflow::expand_dims(input, 0);

    auto output = (*model)({ {MODEL_INPUT, input} }, { MODEL_OUTPUT });
    auto values = output[0].get_data<float>();


    //output = cppflow::cast(output, TF_FLOAT, TF_INT32);
    auto shape2 = output[0].shape();
    auto shape2vec = shape2.get_tensor();
    auto shape2data = shape2.get_data<int64_t>();

    return values;
}

void Transformer::recreate(vector<float> mel) {
    int i;
    //* Denormalize (in place)
    for (i = 0; i < mel.size(); i++) {
        mel[i] = expf32(mel[i]);
    }
    // Convert to matrix for further processing
    auto amp = vecToMat(mel, mel.size() / config.nMel);
    amp.transposeInPlace();

    //* MEL to STFT

    auto basis = (MatrixXf)librosa::internal::melfilter(
        config.sampleRate,
        config.nFFT,
        config.nMel,
        config.fMin,
        config.fMax
    );

    MatrixXf inverse(basis.cols(), amp.cols());
    VectorXf temp(amp.rows());

    nnls_mc<MatrixXf>(
        basis.data(),
        amp.data(),
        inverse.data(),
        (int)basis.rows(),
        (int)basis.cols(),
        (int)amp.cols()
        );

    /*
no instance of constructor "Eigen::NNLS<_MatrixType>::NNLS [with _MatrixType=Eigen::MatrixXf]" matches the argument list -- argument types are: (Eigen::MatrixXf *, Eigen::MatrixXf *, Eigen::MatrixXf *, int, int, int)

    // NNLS
    NNLS<MatrixXf> nnls(basis, 64);

    for (i = 70; i < amp.cols(); i++) {
        temp = amp.col(i);
        if (!nnls.solve(temp)) {
            cout << "Failed to converge on row " << i << endl;
        }
        //NNLS<MatrixXf> nnls(basis);
        inverse.col(i) = nnls.x();
    }

    */
    /*cout << basis.bdcSvd(ComputeThinU | ComputeThinV).solve(mat);

    */


    // Apply inverse exponent (in current case do nothing)



    for (int i = 0; i < 30; i++) {
        cout << mel[i] << " ";
    }
    cout << endl;


    //* Griffin-Lim
    return;
}

