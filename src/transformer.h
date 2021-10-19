#include <stdio.h>
#include <string>
#include <string.h>
#include <codecvt>
#include <locale>
#include <algorithm>
#include <iostream>
#include <map>
#include <vector>

#include <espeak-ng/speak_lib.h>
#include "cppflow/cppflow.h"

#define MODEL_INPUT "serving_default_input_1:0"
#define MODEL_OUTPUT "StatefulPartitionedCall:14"
#define MEL_CHANNELS 80

#define ALL_PHONEMES " !'(),-.:;?abcdefhijklmnopqrstuvwxyzæçðøħŋœǀǁǂǃɐɑɒɓɔɕɖɗɘəɚɛɜɞɟɠɡɢɣɤɥɦɧɨɪɫɬɭɮɯɰɱɲɳɴɵɶɸɹɺɻɽɾʀʁʂʃʄʈʉʊʋʌʍʎʏʐʑʒʔʕʘʙʛʜʝʟʡʢˈˌːˑ˞βθχᵻⱱ"
#define PUNCTUATION "!,-.:;?()"


class Transformer {
public:
    std::string error;
    Transformer(std::string language, std::string espeak_data_path, std::string model_path);
    std::vector<float> Synthesize(std::string text);
private:
    std::map<wchar_t, int> tokenMap;
    cppflow::model* model;

    std::wstring phonemize(std::string text);
    std::vector<int> tokenize(std::wstring phons);
    std::vector<float> runModel(std::vector<int> tokens);
};