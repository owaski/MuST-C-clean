import os
import re
import csv
import argparse
from dataclasses import dataclass

import torch
import torchaudio
import pandas as pd
import editdistance
import spacy
from tqdm import tqdm
from num2words import num2words


def load_df_from_tsv(path: str):
    return pd.read_csv(
        path,
        sep="\t",
        header=0,
        encoding="utf-8",
        escapechar="\\",
        quoting=csv.QUOTE_NONE,
        na_filter=False,
    )

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


def main(args):
    device = torch.device(args.device)

    bundle = torchaudio.pipelines.WAV2VEC2_ASR_LARGE_LV60K_960H
    model = bundle.get_model().to(device)
    labels = bundle.get_labels()
    dictionary = {c: i for i, c in enumerate(labels)}

    mustc_root = args.mustc_root
    df = load_df_from_tsv(args.tsv_path)

    decoder = GreedyCTCDecoder(
        labels=bundle.get_labels(),
        ignore=(0, 1, 2, 3),
    )
    
    mismatch_df = pd.DataFrame(columns=df.columns)
    iterator = tqdm(df.iterrows(), total=df.shape[0], desc='0 mismatch found')
    for _, row in iterator:
        try:
            splits = row['audio'].split(':')
            ori_start, ori_duration = splits[-2:]
            ori_start, ori_duration = int(ori_start), int(ori_duration)
            
            ext = 1

            wav_file = os.path.join(mustc_root, ''.join(splits[:-2]))
            with torch.inference_mode():
                waveform, _ = torchaudio.load(wav_file, \
                    frame_offset=max(ori_start - int(ext * 16000), 0), num_frames=ori_duration + 2 * ext * 16000)
                emissions, _ = model(waveform.to(device))
                emissions = torch.log_softmax(emissions, dim=-1)
            emission = emissions[0].cpu().detach()

            transcript = clean(row['src_text'], dictionary)
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
            flag = start < 0.85 * ext * 16000 or end > ext_len - 0.85 * ext * 16000

            if not flag:
                with torch.inference_mode():
                    waveform, _ = torchaudio.load(wav_file, frame_offset=ori_start, num_frames=ori_duration)
                    emissions, _ = model(waveform.to(device))
                    emissions = torch.log_softmax(emissions, dim=-1)
                emission = emissions[0].cpu().detach()
                asr_transcript = decoder(emission)
                edit_distance = editdistance.eval(transcript, asr_transcript)

                flag |= ext * 16000 < start and ext_len - ext * 16000 > end and \
                    ((ext + 1) * 16000 < start or ext_len - (ext + 1) * 16000 > end) and edit_distance / len(transcript) > 0.3

                flag |= edit_distance / len(transcript) > 0.7

                if flag:
                    print(transcript, asr_transcript, sep='\n')

            if flag:
                mismatch_df = mismatch_df.append(row, ignore_index=True)
                iterator.set_description('{} mismatch found'.format(mismatch_df.shape[0]))
        except Exception as e:
            print(e)

    save_df_to_tsv(mismatch_df, 'mismatch.tsv')
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu', \
        help='device to run detector')
    parser.add_argument('--mustc-root', type=str, required=True, \
        help='directory of must-c dataset')
    parser.add_argument('--tsv-path', type=str, required=True, \
        help='path to MuST-C tsv file path')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)