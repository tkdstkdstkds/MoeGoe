
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool

import operator
import threading
import time
from models import SynthesizerTrn
from torch import no_grad, LongTensor, FloatTensor

import utils
import csv
import MoeGoe
from scipy.io.wavfile import write
import os
import argparse

default_model_path = "BatchConvertModels/"
batch_convert_result_path = "BatchConvertResult/"

def ttsGenerate(csv_row, voice_config_map, synthesizerTrn_map):
    voice_config = voice_config_map[csv_row['voice_config']]
    synthesizerTrn = synthesizerTrn_map[csv_row['voice_config']]

    # check speacker index
    speakerIndex = voice_config.speakers.index(csv_row['voice_character'])
    
    with no_grad():
        # convert text to tensor
        stn_tst = MoeGoe.get_text(f"[JA]{csv_row['text']}[JA]", voice_config, False)
        
        # csv_row['voice_speed'] to float
        voice_speed = 1.0 / float(csv_row['voice_speed'])
        # csv_row['voice_noise'] to float
        voice_noise = float(csv_row['voice_noise'])
        # csv_row['voice_noise_w'] to float 
        voice_noise_w = float(csv_row['voice_noise_w'])

        x_tst = stn_tst.unsqueeze(0)
        x_tst_lengths = LongTensor([stn_tst.size(0)])
        sid = LongTensor([speakerIndex])
        audio = synthesizerTrn.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=0.6, noise_scale_w=0.8,
                            length_scale=voice_speed)[0][0, 0].data.cpu().float().numpy()
    del stn_tst, x_tst, x_tst_lengths, sid

    # save audio
    write(f"{batch_convert_result_path}{csv_row['voice_file']}", voice_config.data.sampling_rate, audio)
    print(f"id:{csv_row['id']} - {csv_row['voice_file']} generated.")

def main(args):
    # load csv file and path = 'test.csv'
    csv_file = open(args.csv_path, encoding='utf-8', mode='r')

    # print csv_file fieldnames
    csv_reader = csv.DictReader(csv_file)

    # using lambda to filter rows which column name 'voice_config' is not empty
    filter_rows = list(filter(lambda row: row['voice_config'].strip(), csv_reader))

    # using set to distinct filter_rows column name 'voice_config'
    distinct_voice_config_path = set(map(lambda row: row['voice_config'].strip(), filter_rows))

    # load all SynthesizerTrn
    voice_config_map = {}
    synthesizerTrn_map = {}
    
    for voice_config_path in distinct_voice_config_path:

        voice_config = utils.get_hparams_from_file(f"{default_model_path}{voice_config_path}.json")
        speakersCount = voice_config.data.n_speakers if 'n_speakers' in voice_config.data.keys() else 0
        symbolsCount = len(voice_config.symbols) if 'symbols' in voice_config.keys() else 0

        synthesizerTrn = SynthesizerTrn(
            symbolsCount,
            voice_config.data.filter_length // 2 + 1,
            voice_config.train.segment_size // voice_config.data.hop_length,
            n_speakers=speakersCount,
            emotion_embedding=False,
            **voice_config.model)
        synthesizerTrn.eval()

        # load checkpoint
        utils.load_checkpoint(f"{default_model_path}{voice_config_path}.pth", synthesizerTrn)

        # add to map
        voice_config_map[voice_config_path] = voice_config
        synthesizerTrn_map[voice_config_path] = synthesizerTrn

    # measure the time
    start_time = time.time()

    # use multi-hreading to generate tts
    # I try to use multi-processing, but it will cause
    # Cowardly refusing to serialize non-leaf tensor which requires_grad, since autograd does not support crossing process boundaries.
    with ThreadPool(args.thread_count) as pool:
        # prepare multiprocessing args
        args = []
        for csv_row in filter_rows:
            args.append((csv_row, voice_config_map, synthesizerTrn_map))
        # start multi-processing
        pool.starmap(ttsGenerate, args)

    # end measure
    end_time = time.time()
    # print total time, time format is second and point two decimal places
    print(f"Total time: {end_time - start_time:.2f} seconds")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Python Command Line Argument Parser')
    # add --csv_path argument
    parser.add_argument('--csv_path', type=str, required=True, help='csv file path')
    # add --thread_count argument
    parser.add_argument('--thread_count', type=int, default=100, help='thread count')
    args = parser.parse_args()

    main(args)

