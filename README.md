# MuST-C-clean
This is the repo for paper "On the Impact of Noises in Crowd-Sourced Data for Speech Translation" in IWSLT 2022. It is currently under development.

This detector is adapted from code in https://pytorch.org/tutorials/intermediate/forced_alignment_with_torchaudio_tutorial.html#sphx-glr-intermediate-forced-alignment-with-torchaudio-tutorial-py.

## Prepare Environment

```bash
conda create python=3.8 -n must-c-clean
conda activate must-c-clean

conda install -y pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install -y tqdm pandas 
conda install -y spacy -c conda-forge
python -m spacy download en_core_web_trf

pip install editdistance num2words pyyaml
```

## Run Detection

You can run the detection as follows:
```bash
python detect.py \
    --device {cpu/cuda} \
    --mustc-root {your must-c root directory} \
    --tgt-lang {de/other languages} \
    --split {train/dev/tst-COMMON/tst-HE}
```

The results will be saved in `results/{split}`. The tsv file `mismatch.tsv` contains the description of the detected audio-transcript mismatch cases. The html file `mismatch.html` allows you to listen to the speech and compare it with the given transcript.