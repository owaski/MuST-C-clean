import os
import re
import csv
import argparse
import logging
from dataclasses import dataclass

import torch
import torchaudio
import pandas as pd
import editdistance
import spacy
from tqdm import tqdm
from num2words import num2words
import yaml

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

SAMPLE_RATE = 16000

def save_df_to_tsv(dataframe, path):
    dataframe.to_csv(
        path,
        sep="\t",
        header=True,
        index=False,
        encoding="utf-8",
        escapechar="\\",
        quoting=csv.QUOTE_NONE,
    )

def get_trellis(emission, tokens, blank_id=0):
    num_frame = emission.size(0)
    num_tokens = len(tokens)

    # Trellis has extra diemsions for both time axis and tokens.
    # The extra dim for tokens represents <SoS> (start-of-sentence)
    # The extra dim for time axis is for simplification of the code.
    trellis = torch.full((num_frame + 1, num_tokens + 1), -float("inf"))
    trellis[:, 0] = 0
    for t in range(num_frame):
        trellis[t + 1, 1:] = torch.maximum(
            # Score for staying at the same token
            trellis[t, 1:] + emission[t, blank_id],
            # Score for changing to the next token
            trellis[t, :-1] + emission[t, tokens],
        )
    return trellis

@dataclass
class Point:
    token_index: int
    time_index: int
    score: float

def backtrack(trellis, emission, tokens, blank_id=0):
    # Note:
    # j and t are indices for trellis, which has extra dimensions
    # for time and tokens at the beginning.
    # When referring to time frame index `T` in trellis,
    # the corresponding index in emission is `T-1`.
    # Similarly, when referring to token index `J` in trellis,
    # the corresponding index in transcript is `J-1`.
    j = trellis.size(1) - 1
    t_start = torch.argmax(trellis[:, j]).item()

    path = []
    for t in range(t_start, 0, -1):
        # 1. Figure out if the current position was stay or change
        # Note (again):
        # `emission[J-1]` is the emission at time frame `J` of trellis dimension.
        # Score for token staying the same from time frame J-1 to T.
        stayed = trellis[t - 1, j] + emission[t - 1, blank_id]
        # Score for token changing from C-1 at T-1 to J at T.
        changed = trellis[t - 1, j - 1] + emission[t - 1, tokens[j - 1]]

        # 2. Store the path with frame-wise probability.
        prob = emission[t - 1, tokens[j - 1] if changed > stayed else 0].exp().item()
        # Return token index and time index in non-trellis coordinate.
        path.append(Point(j - 1, t - 1, prob))

        # 3. Update the token
        if changed > stayed:
            j -= 1
            if j == 0:
                break
    else:
        raise ValueError("Failed to align")
    return path[::-1]


@dataclass
class Segment:
    label: str
    start: int
    end: int
    score: float

    def __repr__(self):
        return f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"

    @property
    def length(self):
        return self.end - self.start


def merge_repeats(path, transcript):
    i1, i2 = 0, 0
    segments = []
    while i1 < len(path):
        while i2 < len(path) and path[i1].token_index == path[i2].token_index:
            i2 += 1
        score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
        segments.append(
            Segment(
                transcript[path[i1].token_index],
                path[i1].time_index,
                path[i2 - 1].time_index + 1,
                score,
            )
        )
        i1 = i2
    return segments

def merge_words(segments, separator="|"):
    words = []
    i1, i2 = 0, 0
    while i1 < len(segments):
        if i2 >= len(segments) or segments[i2].label == separator:
            if i1 != i2:
                segs = segments[i1:i2]
                word = "".join([seg.label for seg in segs])
                score = sum(seg.score * seg.length for seg in segs) / sum(seg.length for seg in segs)
                words.append(Segment(word, segments[i1].start, segments[i2 - 1].end, score))
            i1 = i2 + 1
            i2 = i1
        else:
            i2 += 1
    return words

# noises = ['(Applause)', '(Audience)', '(Audio)', '(Beat)', '(Beatboxing)', '(Beep)', '(Beeps)', '(Cheering)', '(Cheers)', '(Claps)', '(Clicking)', '(Clunk)', '(Coughs)', \
#     '(Drums)', '(Explosion)', '(Gasps)', '(Guitar)', '(Honk)', '(Laugher)', '(Laughing)', '(Laughs)', '(Laughter)', '(Mumbling)', '(Music)', '(Noise)', '(Recording)', \
#     '(Ringing)', '(Shouts)', '(Sigh)', '(Sighs)', '(Silence)', '(Singing)', '(Sings)', '(Spanish)', '(Static)', '(Tones)', '(Trumpet)', '(Video)', '(Voice-over)', \
#     '(Whistle)', '(Whistling)', '(video)']


nlp = spacy.load("en_core_web_trf")
noises = ['(Applause)', '(Audience)', '(Audio)', '(Beat)', '(Beatboxing)', '(Beep)', '(Beeps)', '(Cheering)', '(Cheers)', '(Claps)', '(Clicking)', '(Clunk)', '(Coughs)', \
    '(Drums)', '(Explosion)', '(Gasps)', '(Guitar)', '(Honk)', '(Laugher)', '(Laughing)', '(Laughs)', '(Laughter)', '(Mumbling)', '(Music)', '(Noise)', '(Recording)', \
    '(Ringing)', '(Shouts)', '(Sigh)', '(Sighs)', '(Silence)', '(Singing)', '(Sings)', '(Spanish)', '(Static)', '(Tones)', '(Trumpet)', '(Video)', '(Voice-over)', \
    '(Whistle)', '(Whistling)', '(video)']
def clean(text, dictionary):
    prefix = re.match("(.{,20}:).*", text)

    if prefix is not None:
        prefix_ents = nlp(prefix.group(0)).ents
        if len(prefix_ents) > 0 and prefix_ents[0] == 'PERSON':
            text = text[len(prefix.group(1)):]
    for noise in noises:
        text = text.replace(noise, '')
    tokens = []
    i = 0
    while i < len(text):
        c = text[i]
        if c.isalpha():
            if c.upper() in dictionary:
                tokens.append(c.upper())
        elif c == "'":
            tokens.append(c)
        elif c.isnumeric():
            j = i
            while j + 1 < len(text) and \
                (text[j + 1].isnumeric() or ((text[j + 1] == ',' or text[j + 1] == '.') and (j + 2) < len(text) and text[j + 2].isnumeric())):
                    j += 1
            words = num2words(text[i : j + 1].replace(',', '')).replace(',', '').replace('-', ' ').replace(' ', '|')
            tokens.append(words.upper())
            i = j
        else:
            tokens.append('|')
        i += 1
    transcript = []
    for c in tokens:
        if c == '|':
            if len(transcript) > 0 and transcript[-1] != '|':
                transcript.append(c)
        else:
            transcript.append(c)
    if len(transcript) > 0 and transcript[-1] == '|':
        transcript.pop()
    
    return ''.join(transcript)

class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, ignore):
        super().__init__()
        self.labels = labels
        self.ignore = ignore

    def forward(self, emission: torch.Tensor) -> str:
        """Given a sequence emission over labels, get the best path string
        Args:
        emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
        str: The resulting transcript
        """
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i not in self.ignore]
        return ''.join([self.labels[i] for i in indices])

def write_html(mismatch_df, tgt_lang, split):
    string = '<table>\n'
    string += '\t<tr>\n\t\t<th>Transcript</th>\n\t\t<th>Source Audio</th>\n\t</tr>\n'
    for i in tqdm(range(mismatch_df.shape[0])):
        audio_path = 'wav/{}.wav'.format(i)
        string += '\t<tr>\n\t\t<td>{}</td>\n\t\t<td>{}</td>\n\t</tr>\n'.format(
            mismatch_df['transcript'][i],
            '<audio controls><source src="{}" type="audio/wav"></audio>'.format(audio_path)
        )
    string += '</table>'

    with open('results/{}/{}/mismatch.html'.format(tgt_lang, split), 'w') as w:
        w.write(string)

def main(args):
    device = torch.device(args.device)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    tgt_lang = args.tgt_lang
    split = args.split
    
    os.makedirs('results/{}/{}/wav'.format(tgt_lang, split), exist_ok=True)

    logging.info('Loading wav2vec-2.0-large ASR fine-tuned model')
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_LARGE_LV60K_960H
    model = bundle.get_model().to(device)
    labels = bundle.get_labels()
    dictionary = {c: i for i, c in enumerate(labels)}

    mustc_root = args.mustc_root
    split_root = os.path.join(mustc_root, 'en-{}'.format(args.tgt_lang), 'data', split)

    logging.info('Loading transcripts and yaml file')

    transcripts = []
    with open(os.path.join(split_root, 'txt/{}.en'.format(args.split)), 'r') as r:
        transcripts = [line.strip() for line in r.readlines() if line.strip() != '']
    
    with open(os.path.join(split_root, 'txt/{}.yaml'.format(args.split)), 'r') as r:
        segs = yaml.load(r, yaml.CLoader)

    for seg, t in zip(segs, transcripts):
        seg['transcript'] = t

    decoder = GreedyCTCDecoder(
        labels=bundle.get_labels(),
        ignore=(0, 1, 2, 3),
    )
    
    logging.info('Detecting mismatch')

    mismatch_df = pd.DataFrame(columns=['duration', 'offset', 'speaker_id', 'wav', 'transcript'])
    iterator = tqdm(segs, desc='0 mismatch found')
    finished = True
    tot = 40
    for seg in iterator:
        try:
            tot -= 1
            if tot == 0:
                break
            audio_path = os.path.join(split_root, 'wav/{}'.format(seg['wav']))
            audio_info = torchaudio.info(audio_path)

            if audio_info.sample_rate != SAMPLE_RATE:
                print('The sample rate of wav file is not 16000.')
                finished = False
                break

            ori_start, ori_duration = int(seg['offset'] * SAMPLE_RATE), int(seg['duration'] * SAMPLE_RATE)
            
            ext = 1

            with torch.inference_mode():
                waveform, _ = torchaudio.load(audio_path, \
                    frame_offset=max(ori_start - int(ext * SAMPLE_RATE), 0), num_frames=ori_duration + 2 * ext * SAMPLE_RATE)
                emissions, _ = model(waveform.to(device))
                emissions = torch.log_softmax(emissions, dim=-1)
            emission = emissions[0].cpu().detach()

            transcript = clean(seg['transcript'], dictionary)
            if transcript == '':
                continue
            tokens = [dictionary[c] for c in transcript]
            trellis = get_trellis(emission, tokens)
            path = backtrack(trellis, emission, tokens)
            segments = merge_repeats(path, transcript)
            word_segments = merge_words(segments)

            ratio = waveform.size(1) / (trellis.size(0) - 1)
            start = ratio * word_segments[0].start
            end = ratio * word_segments[-1].end

            ext_len = waveform.size(1)
            flag = start < 0.85 * ext * SAMPLE_RATE or end > ext_len - 0.85 * ext * SAMPLE_RATE

            if not flag:
                with torch.inference_mode():
                    waveform, _ = torchaudio.load(audio_path, frame_offset=ori_start, num_frames=ori_duration)
                    emissions, _ = model(waveform.to(device))
                    emissions = torch.log_softmax(emissions, dim=-1)
                emission = emissions[0].cpu().detach()
                asr_transcript = decoder(emission)
                edit_distance = editdistance.eval(transcript, asr_transcript)

                flag |= ext * SAMPLE_RATE < start and ext_len - ext * SAMPLE_RATE > end and \
                    ((ext + 1) * SAMPLE_RATE < start or ext_len - (ext + 1) * SAMPLE_RATE > end) and edit_distance / len(transcript) > 0.3

                flag |= edit_distance / len(transcript) > 0.7

                # if flag:
                #     print(transcript, asr_transcript, sep='\n')

            if flag:
                mismatch_df = mismatch_df.append(seg, ignore_index=True)
                iterator.set_description('{} mismatch found'.format(mismatch_df.shape[0]))
                
                waveform, _ = torchaudio.load(audio_path, frame_offset=ori_start, num_frames=ori_duration)
                torchaudio.save('results/{}/{}/wav/{}.wav'.format(tgt_lang, split, len(mismatch_df) - 1), waveform, sample_rate=16000)

        except Exception as e:
            pass

    if not finished:
        return 
    
    save_df_to_tsv(mismatch_df, os.path.join('results/{}/{}/mismatch.tsv'.format(tgt_lang, split)))
    write_html(mismatch_df, tgt_lang, split)
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu', \
        help='device to run detector')
    parser.add_argument('--mustc-root', type=str, required=True, \
        help='directory of must-c dataset')
    parser.add_argument('--tgt-lang', type=str, required=True, \
        help='target translation language')
    parser.add_argument('--split', type=str, default='train', \
        help='which split to detect mismatch')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)