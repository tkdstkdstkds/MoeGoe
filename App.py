from text import text_to_sequence, _clean_text
from models import SynthesizerTrn
from torch import no_grad, LongTensor, FloatTensor

import MoeGoe

import gradio as gradio
import webbrowser

import utils
import numpy
import audonnx
import os
import librosa
import re

def ttsGenerate(ttsModelFileDialog, sentenceTextArea, isSymbolCheckbox, speakerNameDropdown, vocalSpeedSlider,
                emotionFileDialog, emotionModelFileDialog):
    global ttsModelConfig
    
    speakersCount = ttsModelConfig.data.n_speakers if 'n_speakers' in ttsModelConfig.data.keys() else 0
    symbolsCount = len(ttsModelConfig.symbols) if 'symbols' in ttsModelConfig.keys() else 0

    emotionEnabled = bool(emotionFileDialog.strip())
    synthesizerTrn = SynthesizerTrn(
        symbolsCount,
        ttsModelConfig.data.filter_length // 2 + 1,
        ttsModelConfig.train.segment_size // ttsModelConfig.data.hop_length,
        n_speakers=speakersCount,
        emotion_embedding=emotionEnabled,
        **ttsModelConfig.model)
    synthesizerTrn.eval()

    utils.load_checkpoint(ttsModelFileDialog, synthesizerTrn)

    emotion = None
    if(emotionEnabled):
        # do emotion extracting
        emotionModel = audonnx.load(os.path.dirname(emotionModelFileDialog))
        audio16000, samplingRate = librosa.load(emotionFileDialog, sr=16000, mono=True)
        emotionSamplingResult = emotionModel(audio16000, samplingRate)['hidden_states']
        emotionNpyPath = re.sub(r'\..*$', '', emotionFileDialog)
        numpy.save(emotionNpyPath, emotionSamplingResult.squeeze(0))
        emotion = FloatTensor(emotionSamplingResult)

    # if language is not None:
    #         text = language_marks[language] + text + language_marks[language]
    speakerIndex = ttsModelConfig.speakers.index(speakerNameDropdown)
    
    stn_tst = MoeGoe.get_text(sentenceTextArea, ttsModelConfig, isSymbolCheckbox)
    
    with no_grad():
        x_tst = stn_tst.unsqueeze(0)
        x_tst_lengths = LongTensor([stn_tst.size(0)])
        sid = LongTensor([speakerIndex])
        audio = synthesizerTrn.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8,
                            length_scale=1.0 / vocalSpeedSlider, emotion_embedding=emotion)[0][0, 0].data.cpu().float().numpy()
    del stn_tst, x_tst, x_tst_lengths, sid

    return "Success", (ttsModelConfig.data.sampling_rate, audio)



def onTTSModelConfigChanged(ttsModelConfigFileDialog):
    global ttsModelConfig
    ttsModelConfig = utils.get_hparams_from_file(ttsModelConfigFileDialog.name)
    speakers = ttsModelConfig.speakers if 'speakers' in ttsModelConfig.keys() else ['0']
    symbols = [[s] for s in ttsModelConfig.symbols]
    # render speaker again
    speakerDropDown = gradio.update(choices=speakers, 
                         value=speakers[0],
                         label="Speaker")
    symbolsDataset = gradio.update(samples=symbols)
                                                  
    return speakerDropDown, symbolsDataset

def onSymbolClick(sentenceTextArea, symbolsDataset):
    return gradio.update(value=sentenceTextArea + ttsModelConfig.symbols[symbolsDataset])

def onIsSymbolClick(isSymbolCheckbox, sentenceTextArea, previousSentenceText):
    global ttsModelConfig
    if isSymbolCheckbox:
        newTextValue = _clean_text(sentenceTextArea, ttsModelConfig.data.text_cleaners) 
        return newTextValue, sentenceTextArea
    else:
        return previousSentenceText, previousSentenceText


def main(): 
    app = gradio.Blocks()
    with app:
        with gradio.Tab("Text-to-Speech"):
            with gradio.Row():
                with gradio.Column(): 
                    ttsModelConfigFileDialog = gradio.File(label="select tts config",
                                                    file_types=[".json"])
                    
                    ttsModelFileDialog = gradio.Textbox(label="select tts model, text path by self")
                    
                    
                    
                    sentenceTextArea = gradio.TextArea(label="Text",
                                            placeholder="Type your sentence here",
                                            value="こんにちわ。")
                    isSymbolCheckbox = gradio.Checkbox(label="Is Symbol")

                    speakerNameDropdown = gradio.Dropdown(label="Speaker")
                    
                    symbolsDataset = gradio.Dataset(label="Symbols", 
                                                  components=["text"],
                                                  type="index",
                                                  samples=[])
                    
                    vocalSpeedSlider = gradio.Slider(minimum=0.1, maximum=5, 
                                                    value=1, step=0.1, 
                                                    label="Vocal Speed")
                    
                    emotionFileDialog = gradio.Textbox(label="select emotion mp3, wav, empty won't have emotion effect")
                                                        #, file_types=[".mp3", ".wav"])
                    emotionModelFileDialog = gradio.Textbox(label="select emotion .onnx model, text path by self")
                    
                    # add event
                    ttsModelConfigFileDialog.change(fn=onTTSModelConfigChanged,
                                                    inputs=[ttsModelConfigFileDialog],
                                                    outputs=[speakerNameDropdown, symbolsDataset])
                    
                    symbolsDataset.click(fn=onSymbolClick,
                                          inputs=[sentenceTextArea, symbolsDataset],
                                          outputs=[sentenceTextArea])
                    
                    # should only for record previous sentenceTextArea
                    previousSentenceText = gradio.Variable()
                    isSymbolCheckbox.change(fn=onIsSymbolClick,
                                            inputs=[isSymbolCheckbox, sentenceTextArea, previousSentenceText],
                                            outputs=[sentenceTextArea, previousSentenceText])
                    

                with gradio.Column():
                    processTextbox = gradio.Textbox(label="Process Text")
                    audioOutputPlayer = gradio.Audio(label="Output Audio")
                    generateAudioButton = gradio.Button("Generate!")
                    generateAudioButton.click(
                        fn=ttsGenerate,
                        inputs=[ttsModelFileDialog, sentenceTextArea, isSymbolCheckbox, speakerNameDropdown, vocalSpeedSlider,
                                emotionFileDialog, emotionModelFileDialog], # noqa
                        outputs=[processTextbox, audioOutputPlayer])

    webbrowser.open(f"http://127.0.0.1:{server_port}?__theme=dark")
    app.launch(server_port=server_port)

if __name__ == '__main__':
    # gradio vars
    server_port = 8359

    #tts model config vars
    ttsModelConfig = None

    main()

