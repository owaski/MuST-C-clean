# MuST-C-clean
This is the repo for paper "On the Impact of Noises in Crowd-Sourced Data for Speech Translation" in IWSLT 2022. It is currently under development.

## Prepare Environment

```bash
conda create python=3.8 -n must-c-clean
conda activate must-c-clean

conda install -y pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install -y tqdm pandas 

pip install editdistance num2words
```


