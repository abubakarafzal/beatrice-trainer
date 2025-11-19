---
license: mit
pipeline_tag: audio-to-audio
tags:
- audio
- speech
- voice-conversion
datasets:
- reazon-research/reazonspeech
- dns-challenge
- libritts-r
---

# Beatrice Trainer

A toolkit for training models for [Beatrice 2](https://prj-beatrice.com), a completely free voice conversion VST featuring ultra-low latency, low load, and low capacity.

Beatrice 2 is developed with the following goals:

* Allow comfortable singing while listening to your converted voice
* Accurately reflect the intonation of the input voice in the converted audio, enabling delicate expression
* High naturalness and clarity of converted audio
* Diverse target speakers
* Approximately 50ms latency when converting with the official VST, measured with external recording equipment
* Low load with RTF < 0.2 when running on a single thread on the developer's notebook PC (Intel Core i7-1165G7)
* Capacity of 30MB or less in minimum configuration
* Operation with VST and [VCClient](https://github.com/w-okada/voice-changer)
* Other features (confidential)

## Release Notes

* **2025-08-31**: Beatrice Trainer 2.0.0-rc.0 released.
  * **Please update the [official VST](https://prj-beatrice.com), [VCClient](https://github.com/w-okada/voice-changer), and [beatrice-client](https://github.com/aq2r/beatrice-client) to the latest version. Models generated with the new Trainer will not work with older versions of the official VST, VCClient, and beatrice-client.**
  * Changed RTF target value from 0.25 to 0.2.
  * Changed package manager from Poetry to uv.
  * Added VocalSet to PitchEstimator training data.
  * Raised PitchEstimator output value upper limit from around A5 to around F6.
  * Changed PitchEstimator to not predict voiced/unvoiced.
  * Fixed missing activation function in PitchEstimator architecture.
  * Improved processing efficiency by adding self-attention and removing GRU in PhoneExtractor architecture.
  * Improved speaker similarity by adding cross-attention structure to inject speaker characteristics in WaveGenerator architecture.
  * Improved generated audio quality by adding noise to PhoneExtractor output during training.
  * Improved speaker similarity by adding vector quantization processing similar to [kNN-VC](https://arxiv.org/abs/2305.18975) to PhoneExtractor output.
  * Improved training stability by adding fine noise to waveforms input to Discriminator.
  * Removed GradientEqualizer as no contribution to quality was confirmed.
  * Improved speaker similarity by adding formant shift to data augmentation processing.
  * Fixed half-frame offset in aperiodicity loss calculation.
  * Improved training stability by setting aperiodicity loss to 0 in parts with very small volume.
  * Improved generated audio quality by adding loudness loss.
  * Changed learning rate scheduling from cosine to exponential, making it easier to extend training.
  * Changed to save checkpoint files compressed.
  * Added configurable items in config file.
  * Disabled TensorBoard numerical recording by default to avoid misunderstanding that quality can be evaluated by loss function values.
  * Adjusted hyperparameters and made several other changes.
* **2024-10-20**: Beatrice Trainer 2.0.0-beta.2 released.
  * **Please update the [official VST](https://prj-beatrice.com) and [VCClient](https://github.com/w-okada/voice-changer) to the latest version. Models generated with the new Trainer will not work with older versions of the official VST and VCClient.**
  * Improved training stability by introducing [Scaled Weight Standardization](https://arxiv.org/abs/2101.08692).
  * Fixed issue where loss calculation results became nan for audio very close to silence, improving training stability.
  * Changed periodic signal generation method, enabling generation of high-quality converted audio with fewer training steps when not using pre-trained models.
  * Improved converted audio quality by introducing post-filter structure inspired by [FIRNet](https://ast-astrec.nict.go.jp/release/preprints/preprint_icassp_2024_ohtani.pdf).
  * Improved converted audio quality by introducing [D4C](https://www.sciencedirect.com/science/article/pii/S0167639316300413) in loss function.
  * Introduced [Multi-scale mel loss](https://arxiv.org/abs/2306.06546).
  * Improved training speed by removing redundant backpropagation and partially disabling `torch.backends.cudnn.benchmark`.
  * Fixed error that occurred when training data contained non-mono audio files.
  * Fixed volume calculation error, resolving inconsistency between training and inference conversion results.
  * Fixed PyTorch version lower limit.
  * Fixed issue where CPU version of PyTorch was installed in Windows environment.
  * Fixed issue where DataLoader operation was very slow in Windows environment.
  * Made several other changes.
* **2024-07-27**: Beatrice Trainer 2.0.0-beta.0 released.

## Prerequisites

Beatrice does not require a GPU to convert voice quality using an existing trained model, but it does require a GPU to efficiently create new models.

When you run the training script, it consumes approximately 9GB of VRAM with the default settings. On a GeForce RTX 4090, training takes approximately 40 minutes to complete.

If you don't have a GPU at hand, you can still train on Google Colab using the following repository:

* Use the included `beatrice_trainer_colab.ipynb` notebook for Google Colab training with T4 GPU support.

## Getting Started

### 1. Download This Repo

Download this repository using Git or similar.

```sh
git lfs install
git clone https://github.com/abubakarafzal/beatrice-trainer.git
cd beatrice-trainer
```

### 2. Environment Setup

Use uv or similar to install the dependent libraries.

```sh
uv sync --extra cu128
. .venv/bin/activate
# Alternatively, you can use pip to install dependencies directly:
# pip3 install -e .[cu128]
```

In a Windows environment, run `.venv\Scripts\activate` instead of `. .venv/bin/activate`.

If the installation is correct, `python3 beatrice_trainer -h` will display the following help:

```
usage: beatrice_trainer [-h] [-d DATA_DIR] [-o OUT_DIR] [-r] [-c CONFIG]

options:
  -h, --help            show this help message and exit
  -d DATA_DIR, --data_dir DATA_DIR
                        directory containing the training data
  -o OUT_DIR, --out_dir OUT_DIR
                        output directory
  -r, --resume          resume training
  -c CONFIG, --config CONFIG
                        path to the config file
```

### 3. Prepare Your Training Data

Arrange the training data as shown below.

```
your_training_data_dir
+---alice
|   +---alices_wonderful_speech.wav
|   +---alices_excellent_speech.flac // FLAC, MP3, and some other formats are also okay.
|   `---...
+---bob
|   +---bobs_fantastic_speech.wav
|   +---bobs_speeches
|   |   `---bobs_awesome_speech.wav // Audio files in nested directory will also be used.
|   `---...
`---...
```

You need to create a directory for each speaker directly under the training data directory. You can freely choose the structure of each speaker's directory and the names of the audio files.

Note that you need to create a speaker directory even if you are training on data for only one speaker.

```
your_training_data_dir_with_only_one_speaker
+---charlies_brilliant_speech.wav // Wrong.
`---...
```

```
your_training_data_dir_with_only_one_speaker
`---charlie
    +---charlies_brilliant_speech.wav // Correct!
    `---...
```

**Example Setup**: This repository includes a sample training setup with pre-configured data and settings:
- Training data: `datasets/model_1/my/` (contains speaker audio files)
- Configuration: `datasets/model_1_config_lowmem.json` (T4 GPU optimized settings with batch_size: 1, hidden_channels: 96)

### 4. Train Your Model

Start training by specifying the directory containing the training data and the output directory.

```sh
python3 beatrice_trainer -d <your_training_data_dir> -o <output_dir>
```

(It has been reported that in Windows, it will not work correctly unless you specify `.\beatrice_trainer\__main__.py` instead of `beatrice_trainer`.)

You can check the training status on TensorBoard.

```sh
tensorboard --logdir <output_dir>
```

**Using the Included Configuration**: To use the included model_1 setup with the pre-configured low-memory settings:

```sh
python3 beatrice_trainer -d datasets/model_1 -o outputs/model_1 -c datasets/model_1_config_lowmem.json
```

This configuration is optimized for T4 GPUs (16GB VRAM) with reduced batch size and hidden channels.

### 5. After Training

When training is completed successfully, a directory named `paraphernalia_(data_dir_name)_(step)` will be generated in the output directory. You can load this directory with the [official VST](https://prj-beatrice.com), [VCClient](https://github.com/w-okada/voice-changer), or [beatrice-client](https://github.com/aq2r/beatrice-client) to perform stream (real-time) conversion. **If you are unable to load this directory, your official VST, VCClient, or beatrice-client may be outdated; please update to the latest version.**

## Detailed Usage

### Training

If you want to use hyperparameters or pre-trained models that are different from the defaults, copy the config file with the default values `assets/default_config.json` to another location, edit the values, and specify the file with `-c`. Please note that editing `assets/default_config.json` directly will break the files.

You can also omit specifying the command line arguments by adding `data_dir` and `out_dir` keys to the configuration file and specifying the directory where the training data is placed and the output directory using absolute paths or paths relative to the repository root.

**Note**: The included `datasets/model_1_config_lowmem.json` is already configured for T4 GPU training and can be used as a reference for low-memory setups.

```sh
python3 beatrice_trainer -c <your_config.json>
```

If training is interrupted for any reason, and the output directory contains `checkpoint_latest.pt.gz`, you can resume training from the last saved checkpoint by adding the `-r` option to the command used for training and executing it.

```sh
python3 beatrice_trainer -d <your_training_data_dir> -o <output_dir> -r
```

For the included model_1 setup:

```sh
python3 beatrice_trainer -d datasets/model_1 -o outputs/model_1 -c datasets/model_1_config_lowmem.json -r
```

### Output Files

When you run the training script, the following files and directories will be generated in the output directory.

* `paraphernalia_(data_dir_name)_(step)`
  * A directory containing all the files required for stream conversion.
  * Some learning steps may be output, so it is fine to delete any steps other than those required.
  * Output files outside this directory will not be used for stream conversion, so you can safely delete them if you do not need them.
* `checkpoint_(data_dir_name)_(step).pt.gz`
  * This is a checkpoint to resume your learning from where you left off.
  * If you rename it to `checkpoint_latest.pt.gz` and run the training script with the `-r` option, you can resume training from that step number.
* `checkpoint_latest.pt.gz`
  * A copy of the most recent `checkpoint_(data_dir_name)_(step).pt.gz`.
* `config.json`
  * This is the configuration used for training.
* `events.out.tfevents.*`
  * This data contains the information displayed in TensorBoard.

### Customize Paraphernalia

You can change the display on VST, VCClient, and beatrice-client by editing the `beatrice_paraphernalia_*.toml` files in the paraphernalia directory generated by the learning script.

`model.version` represents the format version of the generated model and should not be changed.

If each `description` line is too long, the entire text may not be displayed. Even if it can be displayed now, it may become impossible to display due to future changes in the VST, VCClient, or beatrice-client specifications, so please limit the number of characters and lines to a reasonable number.

The image you set for `portrait` must be in PNG format and square.

## Distribution of Trained Models

You are welcome to distribute models generated using this repository.

The distributed models may be introduced on social media accounts and websites managed by Project Beatrice and its affiliates. Please be aware that in such cases, images set in `portrait` may be posted.

## Resource

This repository contains various data used for training etc. Please see [assets/README.md](https://huggingface.co/fierce-cats/beatrice-trainer/blob/main/assets/README.md) for details.

## Reference

* [wav2vec 2.0](https://arxiv.org/abs/2006.11477) ([Official implementation](https://github.com/facebookresearch/fairseq), [MIT License](https://github.com/facebookresearch/fairseq/blob/main/LICENSE))
  * Used in FeatureExtractor implementation.
* [EnCodec](https://arxiv.org/abs/2210.13438) ([Official implementation](https://github.com/facebookresearch/encodec), [MIT License](https://github.com/facebookresearch/encodec/blob/main/LICENSE))
  * Used in GradBalancer implementation.
* [HiFi-GAN](https://arxiv.org/abs/2010.05646) ([Official implementation](https://github.com/jik876/hifi-gan), [MIT License](https://github.com/jik876/hifi-gan/blob/master/LICENSE))
  * Used in DiscriminatorP implementation.
* [Vocos](https://arxiv.org/abs/2306.00814) ([Official implementation](https://github.com/gemelo-ai/vocos), [MIT License](https://github.com/gemelo-ai/vocos/blob/main/LICENSE))
  * Used in ConvNeXtBlock implementation.
* [BigVSAN](https://arxiv.org/abs/2309.02836) ([Official implementation](https://github.com/sony/bigvsan), [MIT License](https://github.com/sony/bigvsan/blob/main/LICENSE))
  * Used in SAN module implementation.
* [D4C](https://www.sciencedirect.com/science/article/pii/S0167639316300413) ([Unofficial implementation by tuanad121](https://github.com/tuanad121/Python-WORLD), [MIT License](https://github.com/tuanad121/Python-WORLD/blob/master/LICENSE.txt))
  * Used in loss function implementation.
* [UnivNet](https://arxiv.org/abs/2106.07889) ([Unofficial implementation by maum-ai](https://github.com/maum-ai/univnet), [BSD 3-Clause License](https://github.com/maum-ai/univnet/blob/master/LICENSE))
  * Used in DiscriminatorR implementation.
* [FragmentVC](https://arxiv.org/abs/2010.14150)
  * Used idea of injecting voice quality using cross-attention with features derived from SSL model as query.
* [NF-ResNets](https://arxiv.org/abs/2101.08692)
  * Used idea of Scaled Weight Standardization.
* [Soft-VC](https://arxiv.org/abs/2111.02392)
  * Used as basic idea for PhoneExtractor.
* [kNN-VC](https://arxiv.org/abs/2305.18975)
  * Used idea of voice conversion scheme as auxiliary.
* [Descript Audio Codec](https://arxiv.org/abs/2306.06546)
  * Used idea of Multi-scale mel loss.
* [StreamVC](https://arxiv.org/abs/2401.03078)
  * Used as basic idea for voice conversion scheme.
* [FIRNet](https://ast-astrec.nict.go.jp/release/preprints/preprint_icassp_2024_ohtani.pdf)
  * Used idea of applying FIR filter to vocoder.
* [EVA-GAN](https://arxiv.org/abs/2402.00892)
  * Used idea of applying SiLU to vocoder.
* [Subramani et al., 2024](https://arxiv.org/abs/2309.14507)
  * Used as basic idea for PitchEstimator.
* [Agrawal et al., 2024](https://arxiv.org/abs/2401.10460)
  * Used as basic idea for Vocoder.

## License

The source code and pre-trained models in this repository are published under the MIT License. Please see [LICENSE](https://huggingface.co/fierce-cats/beatrice-trainer/blob/main/LICENSE) for details.
