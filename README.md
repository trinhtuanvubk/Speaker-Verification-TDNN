# Introduction
<!-- https://github.com/konas122/Voiceprint-Recognition -->


## Data Structure


```
.
|___data
│   |___train
        |___speaker1
            |___audio1.wav
            |___ ....
            |___audion.wav
        |___ ....
        |___speakern
            |___audio1.wav
            |___ ....
            |___audion.wav
│   ├── val
│   └── test
```

NOTE: The original repo has something wrong when splitting data, you should put all data on train folder (and a small part on val and test)


## Training
- Download pretrained model at [param.model](https://github.com/konas122/tdnn-on-directml/releases/download/v1.0/param.model)

- To finetune, run:
```bash
python3 main.py --scenario train --load_pretrained
```

- To train, run:
```bash
python3 main.py --scenario train
```


## Reference

Original ECAPA-TDNN paper
```
@inproceedings{desplanques2020ecapa,
  title={{ECAPA-TDNN: Emphasized Channel Attention, propagation and aggregation in TDNN based speaker verification}},
  author={Desplanques, Brecht and Thienpondt, Jenthe and Demuynck, Kris},
  booktitle={Interspeech 2020},
  pages={3830--3834},
  year={2020}
}
```


## Acknowledge

We study many useful projects in our codeing process, which includes:

[Ecapa-tdnn: Emphasized channel attention, propagation and aggregation in tdnn based speaker verification.](https://arxiv.org/abs/2005.07143v3)

[clovaai/voxceleb_trainer](https://github.com/clovaai/voxceleb_trainer).

[lawlict/ECAPA-TDNN](https://github.com/lawlict/ECAPA-TDNN/blob/master/ecapa_tdnn.py).

[TaoRuijie/ECAPA-TDNN](https://github.com/TaoRuijie/ECAPA-TDNN)

Thanks for these authors to open source their code!
