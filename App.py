from text import text_to_sequence, _clean_text
from models import SynthesizerTrn
from torch import no_grad, LongTensor

import MoeGoe

import gradio as gradio
import webbrowser
import torch
import utils




def ttsGenerate(ttsModelFileDialog, sentenceTextArea, isSymbolCheckbox, speakerNameDropdown, vocalSpeedSlider):
    global ttsModelConfig
    
    speakersCount = ttsModelConfig.data.n_speakers if 'n_speakers' in ttsModelConfig.data.keys() else 0
    symbolsCount = len(ttsModelConfig.symbols) if 'symbols' in ttsModelConfig.keys() else 0

    synthesizerTrn = SynthesizerTrn(
        symbolsCount,
        ttsModelConfig.data.filter_length // 2 + 1,
        ttsModelConfig.train.segment_size // ttsModelConfig.data.hop_length,
        n_speakers=speakersCount,
        emotion_embedding=False,
        **ttsModelConfig.model)
    synthesizerTrn.eval()

    utils.load_checkpoint(ttsModelFileDialog, synthesizerTrn)

    # if language is not None:
    #         text = language_marks[language] + text + language_marks[language]
    speakerIndex = ttsModelConfig.speakers.index(speakerNameDropdown)
    
    stn_tst = MoeGoe.get_text(sentenceTextArea, ttsModelConfig, isSymbolCheckbox)
    
    
    with no_grad():
        x_tst = stn_tst.unsqueeze(0).to(device)
        x_tst_lengths = LongTensor([stn_tst.size(0)]).to(device)
        sid = LongTensor([speakerIndex]).to(device)
        audio = synthesizerTrn.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8,
                            length_scale=1.0 / vocalSpeedSlider)[0][0, 0].data.cpu().float().numpy()
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
                        inputs=[ttsModelFileDialog, sentenceTextArea, isSymbolCheckbox, speakerNameDropdown, vocalSpeedSlider], # noqa
                        outputs=[processTextbox, audioOutputPlayer])

    webbrowser.open(f"http://127.0.0.1:{server_port}")
    app.launch(server_port=server_port)

if __name__ == '__main__':
    # gradio vars
    server_port = 8359

    #TODO: should output which graphic card is available
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    #tts model config vars
    ttsModelConfig = None

    main()

