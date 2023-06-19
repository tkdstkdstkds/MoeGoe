from cProfile import label
from altair import value
from text import cleaners, text_to_sequence, _clean_text
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
import traceback

def ttsGenerate(ttsModelFileDialog, sentenceTextArea, isSymbolCheckbox, speakerNameDropdown, vocalSpeedSlider,
                emotionFileDialog, emotionModelFileDialog, noiseScaleSlider, noiseScaleWidthSlider):
    global ttsModelConfig

    if ttsModelConfig is None:
        return "Missing tts config!", None
    
    if not ttsModelFileDialog.strip():
        return "Missing tts model!", None
    
    if emotionFileDialog is not None and not emotionModelFileDialog.strip():
        return "Missing emotion model", None

    try:
        speakersCount = ttsModelConfig.data.n_speakers if 'n_speakers' in ttsModelConfig.data.keys() else 0
        symbolsCount = len(ttsModelConfig.symbols) if 'symbols' in ttsModelConfig.keys() else 0

        emotionEnabled = emotionFileDialog is not None
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
            audio16000, samplingRate = librosa.load(emotionFileDialog.name, sr=16000, mono=True)
            emotionSamplingResult = emotionModel(audio16000, samplingRate)['hidden_states']
            emotionNpyPath = re.sub(r'\..*$', '', emotionFileDialog.name)
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
            audio = synthesizerTrn.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=noiseScaleSlider, noise_scale_w=noiseScaleWidthSlider,
                                length_scale=1.0 / vocalSpeedSlider, emotion_embedding=emotion)[0][0, 0].data.cpu().float().numpy()
        del stn_tst, x_tst, x_tst_lengths, sid

        return "Success", (ttsModelConfig.data.sampling_rate, audio)
    except Exception:
        stackTrace = traceback.format_exc()
        print(stackTrace)
        return stackTrace, None



def onTTSModelConfigChanged(ttsModelConfigFileDialog):
    global ttsModelConfig
    global cleanersDescription

    if ttsModelConfigFileDialog is None:
        ttsModelConfig = None
        speakerNameDropdown = gradio.update(choices=[], value="")
        symbolsDataset = gradio.update(samples=[])
        cleanersLabel = gradio.update(label="language", value="")
        return speakerNameDropdown, symbolsDataset, cleanersLabel
    
    ttsModelConfig = utils.get_hparams_from_file(ttsModelConfigFileDialog.name)
    speakers = ttsModelConfig.speakers if 'speakers' in ttsModelConfig.keys() else ['0']
    symbols = [[s] for s in ttsModelConfig.symbols]
    # render speaker again
    speakerDropDown = gradio.update(choices=speakers, 
                         value=speakers[0],
                         label="Speaker")
    symbolsDataset = gradio.update(samples=symbols)
    cleanersLabel = gradio.update(label=ttsModelConfig.data.text_cleaners[0],
                                  value=cleanersDescription[ttsModelConfig.data.text_cleaners[0]])
                                                  
    return speakerDropDown, symbolsDataset, cleanersLabel

def onSymbolClick(sentenceTextArea, symbolsDataset):
    global ttsModelConfig

    return gradio.update(value=sentenceTextArea + ttsModelConfig.symbols[symbolsDataset])

def onIsSymbolClick(isSymbolCheckbox, sentenceTextArea, previousSentenceText):
    global ttsModelConfig

    if isSymbolCheckbox:
        newTextValue = _clean_text(sentenceTextArea, ttsModelConfig.data.text_cleaners) 
        return newTextValue, sentenceTextArea
    else:
        return previousSentenceText, previousSentenceText

def onEmotionFileChange(emotionFileDialog):
    if emotionFileDialog is None:
        return gradio.update(value=None, visible=False)
    else:
        return gradio.update(value=emotionFileDialog.name, visible=True)
    
def onIsDefaultEmotionModelCheckbox(isDefaultEmotionModelCheckbox):
    global ttsModelConfig

    if isDefaultEmotionModelCheckbox:
        return gradio.update(visible=False)
    else:
        return gradio.update(visible=True)


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
                    

                    speakerNameDropdown = gradio.Dropdown(label="Speaker")
                    
                    isSymbolCheckbox = gradio.Checkbox(label="Is Symbol")
                    symbolsDataset = gradio.Dataset(label="Symbols", 
                                                  components=["text"],
                                                  type="index",
                                                  samples=[],
                                                  visible=True)
                    with gradio.Row():
                        vocalSpeedSlider = gradio.Slider(minimum=0.1, maximum=5, 
                                                        value=1, step=0.1, 
                                                        label="Vocal Speed")
                        
                        noiseScaleSlider = gradio.Slider(minimum=0.1, maximum=3, 
                                                        value=0.667, step=0.1, 
                                                        label="Noise Scale")
                        noiseScaleWidthSlider = gradio.Slider(minimum=0.1, maximum=3, 
                                                        value=0.8, step=0.1, 
                                                        label="Stretched Vocal")
                    
                    with gradio.Row():
                        emotionFileDialog = gradio.File(label="select emotion mp3, wav, empty won't have emotion effect",
                                                        file_types=[".mp3", ".wav"])
                        emotionAudioPlayer = gradio.Audio(visible=False)
                        emotionFileDialog.change(fn=onEmotionFileChange,
                                                 inputs=[emotionFileDialog],
                                                 outputs=[emotionAudioPlayer])
                    
                    with gradio.Row():
                        isDefaultEmotionModelCheckbox = gradio.Checkbox(label="Use default Emotion Model", value=True)
                        emotionModelFileDialog = gradio.Textbox(label="select emotion .onnx model, text path by self",
                                                                value="./emotionModel/model.onnx",
                                                                visible=False)
                        isDefaultEmotionModelCheckbox.change(fn=onIsDefaultEmotionModelCheckbox,
                                                 inputs=[isDefaultEmotionModelCheckbox],
                                                 outputs=[emotionModelFileDialog])
                        
                    
                    cleanersLabel = gradio.Label(label="language")

                    # add event
                    ttsModelConfigFileDialog.change(fn=onTTSModelConfigChanged,
                                                    inputs=[ttsModelConfigFileDialog],
                                                    outputs=[speakerNameDropdown, symbolsDataset, cleanersLabel])
                    
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
                                emotionFileDialog, emotionModelFileDialog, noiseScaleSlider, noiseScaleWidthSlider], # noqa
                        outputs=[processTextbox, audioOutputPlayer])

    webbrowser.open(f"http://127.0.0.1:{server_port}?__theme=dark")
    app.launch(server_port=server_port)

if __name__ == '__main__':
    # gradio vars
    server_port = 8359

    #tts model config vars
    ttsModelConfig = None

    cleanersDescription = \
    {
        "japanese_cleaners": "only support japanese, sentense don't need any tag",
        "japanese_cleaners2": "only support japanese, sentense don't need any tag",
        "korean_cleaners": "only support korean, sentense don't need any tag",
        "chinese_cleaners": "only support chinese, sentense don't need any tag",
        "zh_ja_mixture_cleaners": "support chinese and japanese, sentense pattern should be [ZH]中文[ZH] [JA]こんにちは[JA]",
        "sanskrit_cleaners": "only support sanskrit, sentense don't need any tag",
        "cjks_cleaners": "support chinese japanese korean and sanskrit, sentense pattern should be [ZH]中文[ZH] [JA]こんにちは[JA] [KO]안녕하세요[KO] [SA]नमस्ते[SA]",
        "cjke_cleaners": "support chinese japanese korean and english, sentense pattern should be [ZH]中文[ZH] [JA]こんにちは[JA] [KO]안녕하세요[KO] [EN]English[EN]",
        "cjke_cleaners2": "support chinese japanese korean and english, sentense pattern should be [ZH]中文[ZH] [JA]こんにちは[JA] [KO]안녕하세요[KO] [EN]English[EN]",
        "thai_cleaners": "only support thai, don't need any tag",
        "shanghainese_cleaners": "only support shanghainese, don't need any tag",
        
        "chinese_dialect_cleaners": "support chinese_dialects japanese and english, \
        sentense pattern should be 日本語[JA], English[EN], 普通话[ZH], 上海话[SH], 广东话[GD], 苏州话[SZ], 无锡话[WX], 杭州话[HZ], 绍兴话[SX], 宁波话[NB], 靖江话[JJ], 宜兴话[YX], 嘉定话[JD], 真如话[ZR], 平湖话[PH], 桐乡话[TX], 嘉善话[JS], 硖石话[HN], 临平话[LP], 萧山话[XS], 富阳话[FY], 儒嶴话[RA], 慈溪话[CX], 三门话[SM], 天台话[TT], 温州话[WZ], 遂昌话[SC], 游埠话[YB]"
    }

    main()

