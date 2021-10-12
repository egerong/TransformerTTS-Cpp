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

using namespace std;

#define ALL_PHONEMES " !'(),-.:;?abcdefhijklmnopqrstuvwxyzæçðøħŋœǀǁǂǃɐɑɒɓɔɕɖɗɘəɚɛɜɞɟɠɡɢɣɤɥɦɧɨɪɫɬɭɮɯɰɱɲɳɴɵɶɸɹɺɻɽɾʀʁʂʃʄʈʉʊʋʌʍʎʏʐʑʒʔʕʘʙʛʜʝʟʡʢˈˌːˑ˞βθχᵻⱱ"
#define PUNCTUATION "!,-.:;?()"


wstring phonemize(std::string text);
vector<int> tokenize(wstring phons);

using convert_t = std::codecvt_utf8<wchar_t>;
wstring_convert<convert_t, wchar_t> strconverter;

map<wchar_t, int> token_map;

int InitPreprocessor(const char *language, const char *data_path) {
    // Init phonemizer
    int error;
    espeak_Initialize(AUDIO_OUTPUT_SYNCHRONOUS, 1, data_path, 0);
    error = espeak_SetVoiceByName(language);
    switch (error) {
    case 0:
        break;
    default:
        printf("Error setting language: %d\n", error);
        return error;
    }
    // Init tokenizer
    wstring all_phonemes = strconverter.from_bytes(ALL_PHONEMES);
    token_map.insert(pair<wchar_t, int>('/', 0)); // Padding token
    for (int i=0; i < all_phonemes.length(); i++) {
        token_map.insert(pair<wchar_t, int>(all_phonemes[i], i+1));
    }
    
    return 0;
}

vector<int> PreProcess(string text) {
    wstring phons = phonemize(text);
    cout << "Phonemes" << strconverter.to_bytes(phons) << endl;
    vector<int> tokens = tokenize(phons);
    printf("Tokens: ");
    for (int i=0; i<tokens.size(); i++) {
        printf("%d, ", tokens[i]);
    }
    return tokens;
}

wstring phonemize(string text) {
    const char * c_text = text.c_str();
    const char ** text_ptr = &c_text;
    const char * p;
    string phon;
    
    int phonememode = 1;
    phonememode |= espeakPHONEMES_IPA; // Use IPA symbols
    //phonememode |= espeakPHONEMES_TIE; // Use the separator (if set) as a tie
    //phonememode |= ' ' << 8; // Use space as a separator

    const char * next_punct;
    char punct;
    while (*text_ptr) {
        next_punct = strpbrk(*text_ptr, PUNCTUATION);
        p = espeak_TextToPhonemes(
            (const void **)text_ptr, 
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

vector<int> tokenize(wstring phons) {
    vector<int> tokens;
    for (int i=0; i<phons.length(); i++) {
        tokens.push_back(token_map[phons[i]]);
    }
    return tokens;
}