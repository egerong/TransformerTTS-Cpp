#include <stdio.h>
#include <string.h>
#include <iostream>

#include <espeak-ng/speak_lib.h>

#define PUNCTUATION ";:,.!?¡¿—…\"«»“”"

int InitPhonemizer(const char *language, const char *data_path) {
    int error;
    espeak_Initialize(AUDIO_OUTPUT_SYNCHRONOUS, 1, data_path, 0);
    error = espeak_SetVoiceByName(language);
    switch (error) {
    case 0:
        return 0;
    default:
        printf("Error setting language: %d\n", error);
        return error;
    }
}

std::string Phonemize(std::string text) {
    const char * c_text = text.c_str();
    const char ** text_ptr = &c_text;
    const char * phonemes;
    std::string all_phonemes;
    
    int phonememode = 1;
    phonememode |= espeakPHONEMES_IPA; // Use IPA symbols
    //phonememode |= espeakPHONEMES_TIE; // Use the separator (if set) as a tie
    //phonememode |= ' ' << 8; // Use space as a separator

    const char * next_punct;
    char punct;
    while (*text_ptr) {
        next_punct = strpbrk(*text_ptr, PUNCTUATION);
        phonemes = espeak_TextToPhonemes(
            (const void **)text_ptr, 
            espeakCHARS_UTF8, 
            phonememode
        );
        all_phonemes = all_phonemes + phonemes;
        if (next_punct) {
            all_phonemes = all_phonemes + next_punct[0];
        }
        //strcat(all_phonemes, ",");
        //strcat(all_phonemes, phonemes);
    }
    return all_phonemes;
}




    
   
