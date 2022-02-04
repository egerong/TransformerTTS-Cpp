#include "transformer.h"

using namespace std;

#define ALL_PHONEMES " !'(),-.:;?abcdefhijklmnopqrstuvwxyzæçðøħŋœǀǁǂǃɐɑɒɓɔɕɖɗɘəɚɛɜɞɟɠɡɢɣɤɥɦɧɨɪɫɬɭɮɯɰɱɲɳɴɵɶɸɹɺɻɽɾʀʁʂʃʄʈʉʊʋʌʍʎʏʐʑʒʔʕʘʙʛʜʝʟʡʢˈˌːˑ˞βθχᵻⱱ"
#define PUNCTUATION "!,-.:;?()"

// Model input and output names, get with command
// saved_model_cli show --dir model --all
#define MODEL_INPUT "serving_default_input_1:0"
#define MODEL_OUTPUT "StatefulPartitionedCall:12"

#define VOCODER_INPUT "serving_default_mels:0"
#define VOCODER_OUTPUT "StatefulPartitionedCall:0"

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
    // Load vocoder
    vocoder = torch::jit::load("/home/egert/EKI/TransformerTTS/out/hifigan");
}

void Transformer::Synthesize(string text) {

    auto phonemes = phonemize(text);
    auto tokens = tokenize(phonemes);
    auto mel = runModel(tokens);
    vector<float> wav = vocode(mel);
    //MatrixXd s = melToSTFT(mel);
    //vector<float> wav = griffinLim(s);
    bool success = saveWAV("filename", wav);
    //matToCSV(stft, "/home/egert/Prog/TTS-CPP/temp/inverse.csv");
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
    return values;
}

vector<float> Transformer::vocode(vector<float> mel) {


    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    //unsigned long int shape = { config.nMel, mel.size() };
    torch::Tensor inputTensor = torch::from_blob(
        mel.data(),
        { 1, unsigned(mel.size() / config.nMel), config.nMel },
        torch::TensorOptions().dtype(torch::kFloat32)
    );
    inputTensor = inputTensor.transpose(1, 2);

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(inputTensor);

    auto outputs = vocoder.forward(inputs);
    auto outputTensor = outputs.toTensor();
    //vector<float> wavFloat(wavDouble.begin(), wavDouble.end());

    vector<float> wav(
        outputTensor.data_ptr<float>(),
        outputTensor.data_ptr<float>() + outputTensor.numel()
    );
    //scout << outputTensor.size() << endl;
    return wav;
}

void Test(void) {
}




bool Transformer::saveWAV(string filename, vector<float> data) {
    AudioFile<float> a;
    a.setNumChannels(1);
    a.setSampleRate(config.sampleRate);
    //a.setNumSamplesPerChannel(data.size());
    a.samples[0] = data;
    return a.save("/home/egert/EKI/TTS-CPP/test.wav", AudioFileFormat::Wave);
}
