# Announcements

* I uploaded a tutorial video explaining Style-Bert-VITS2, which can be found on [YouTube](https://www.youtube-nocookie.com/embed/aTUSzgDl1iY).

* I frequently visit the「AI声づくり研究会」 (AI Voice Creation Research Group) Discord server.

# Overview

On February 1, 2024, a Japanese-specialized version of the Chinese open-source text-to-speech (TTS) model Bert-VITS2, called [Bert-VITS2 JP-Extra](https://github.com/fishaudio/Bert-VITS2/releases/tag/JP-Exta), was released. My modified version, [Style-Bert-VITS2](https://github.com/litagin02/Style-Bert-VITS2), now supports the JP-Extra version as of February 3.

You can try out the model using the [online demo](https://huggingface.co/spaces/litagin/Style-Bert-VITS2-JVNV).

This version improves the naturalness of Japanese pronunciation, accent, and intonation, while reducing clarity issues and instability during training. If you only need Japanese TTS and don't require English or Chinese, using the JP-Extra version is highly recommended.

This article discusses the differences between the JP-Extra version and the [previous 2.1-2.3 structures](https://zenn.dev/litagin/articles/8c6edcf6b6fcd6), as well as how Style-Bert-VITS2 further modifies the model.

# Disclaimer

I have not formally studied machine learning, voice AI, or Japanese language processing, so there may be inaccuracies in the article.

# Short Summary

Compared to other versions, (Style-)Bert-VITS2 JP-Extra:

- Fixes bugs in the Japanese reading and accent acquisition parts of the original version (thanks to my contributions ✌)
- Increases the amount of Japanese training data used for the pre-trained model (approximately 800 hours for Japanese only)
- Removes Chinese and English components to focus on Japanese performance
- Implements voice control using prompts with CLAP, as in version 2.2 (although it doesn't seem very practical, similar to 2.2)
- Uses the new WavLM-based discriminator from version 2.3 to improve naturalness
- Removes the duration discriminator to avoid unstable phoneme intervals, as seen in version 2.3
- In Style-Bert-VITS2, the seemingly ineffective CLAP-based emotion embedding is removed and replaced with a simple fully connected layer for style embedding, as in the previous version
- Style-Bert-VITS2 also allows for (some) manual control of accents

These changes significantly improve the naturalness of Japanese pronunciation and accents, increase clarity, and reduce the impression of "Japanese spoken by a foreigner" that was present in earlier versions.

Below I will write a little about these changes from a layman's perspective.

# Increase in Japanese Training Data

According to the [Bert-VITS2 JP-Extra release page](https://github.com/fishaudio/Bert-VITS2/releases/tag/JP-Exta); 

> 3. "the amount of Japanese training data has been increased several times, now up to approximately 800 hours for a single language"

The increase in data (along with the fixes to Japanese processing bugs) may contribute more to the improvement in naturalness than the model structure refinements, although this is not certain without further experimentation.

# Japanese Language Processing

The following section is not directly related to the main topic of model structure, so feel free to skip it if you are not interested.

In a previous article, I mentioned that the original version had bugs in the Japanese processing part, and that Style-Bert-VITS2 fixed them. The current (Style-)Bert-VITS2 JP-Extra incorporates those fixes. Here, I will explain precisely what kind of processing is performed on Japanese text.

## Overview of Japanese Processing in Japanese TTS Models

In Japanese TTS, the input Japanese text is typically converted into a phoneme sequence, a process called grapheme-to-phoneme (g2p). Both during training and inference, the phoneme sequence obtained from the g2p process is fed into the model as input (in addition to the original Japanese text via BERT, which is a unique feature of Bert-VITS2).

## Example

The [pyopenjtalk](https://github.com/r9y9/pyopenjtalk) library, which has become a de facto standard for Japanese phoneme processing and is used in Bert-VITS2, provides a g2p function. For example:

```python
>>> import pyopenjtalk
>>> pyopenjtalk.g2p("おはよう！元気ですか？")
'o h a y o o pau g e N k i d e s U k a'
```

The function returns a space-separated list of phonemes for the input text.

## Limitations of pyopenjtalk's Default g2p

While `pyopenjtalk.g2p` is convenient, it has some limitations:

1. It only returns a simple phoneme sequence without any accent information.
2. All punctuation marks and symbols like "！？" in the input text are treated as pause phonemes (`pau`), so "私は……そう思う……。" and "私は！！！！そう思う！！！" are not distinguished.

### g2p Considering Accents

Accents are an important issue in Japanese TTS. Unlike English or Chinese, each word in Japanese has a correct accent, and incorrect accents can cause significant unnaturalness. Therefore, it is desirable to correctly learn the accent information in addition to the phonemes and, if possible, to enable manual accent specification for TTS.

There are various approaches to incorporating accent information into the model. For example, [ESPNet](https://github.com/espnet/espnet), which allows for various voice-related training tasks, provides the [following g2p functions](https://github.com/espnet/espnet/blob/59733c2f1a962575667f6887e87fcdf04e06afc3/egs2/jvs/tts1/run.sh#L29-L49) for Japanese:

- `pyopenjtalk`: Uses the default `pyopenjtalk.g2p` function without accent information
- `pyopenjtalk_accent`: Inserts accent information using `0` for low and `1` for high pitch after each phoneme
- `pyopenjtalk_prosody`: Inserts `[` for pitch rise and `]` for pitch fall as part of the phoneme symbols
- `pyopenjtalk_kana`: Outputs katakana instead of phonemes
- `pyopenjtalk_phone`: Outputs phonemes with stress and tone marks

The choice of which g2p function to use varies among libraries, and it is unclear which one is the most common. For example, [COEIROINK](https://coeiroink.com/) uses `pyopenjtalk_prosody`.

However, **all of these g2p functions have the second limitation mentioned above, where the type and number of symbols in the input text are not distinguished**. We desire the model to read "私は……そう思う……。" with a lack of confidence and "私は！！！！そう思う！！！" as shouting loudly, but this is not possible with these functions.

In Bert-VITS2, accent information is fed into the model separately from the phoneme sequence under the name `tones`. This assigns the numbers 0 or 1 to each phoneme in the phoneme sequence. For Japanese, it looks like this:

```
おはよう！！！ございます？
→ (o: 0), (h: 1), (a: 1), (y: 1), (o: 1), (o: 1), (!, 0), (!, 0), (!, 0), (g: 0) (o: 0), (z: 1) (a: 1), (i: 1), (m: 1), (a: 1), (s: 0), (u: 0), (?, 0)
```

The low (0) and high (1) values are assigned to each phoneme, and this sequence of values is fed into the model separately from the phoneme sequence. Furthermore, as in the example above, **the symbols in the text are treated as phonemes themselves, distinguishing their type and number**.

Implementing such a g2p function might be easy for experts in the Japanese TTS field, but lacking such knowledge, I took the following approach:

1. First, obtain a phoneme sequence with pitch rise and fall symbols using `pyopenjtalk_prosody` from ESPNet (however, the information about symbols like "！" and "…" is lost)
2. Use this to create a list of phoneme-accent pairs with the symbols completely removed
3. Separately create a phoneme sequence (without accent information) that includes the symbols
4. Combine the two results to obtain the desired output

It might be possible to perform these operations using `pyopenjtalk` alone, but there are some difficulties:

- In pyopenjtalk (OpenJTalk), obtaining accent information seems to always require extracting full context labels from the text (`pyopenjtalk.extract_fullcontext`), but the information about the type and number of symbols in the text is already lost at the full context label stage (so it is fundamentally impossible to obtain the desired result by parsing the full context labels)

On the other hand, for step 3 above, the `pyopenjtalk.run_frontend` function obtains the reading in `pron` while preserving the type and number of symbols, as follows:

As you can see, the `pron` part of the `pyopenjtalk.run_frontend` output preserves the type and number of symbols, so converting this to a phoneme sequence would accomplish step 3.

Then, it's just a matter of writing processing code to combine the two results.

For more details (and for my own reference), please refer to the well-commented source code.

I actively welcome suggestions on how to simplify this process.

## Previously Existing Bugs

Previous versions of Bert-VITS2 did not use the above method and had the following bugs:

1. When converting the reading result of `pyopenjtalk.run_frontend` to a phoneme list, the katakana reading was further processed by `pyopenjtalk.g2p`, causing issues.
2. The accent acquisition method was inaccurate, and the information was reset at word boundaries. For example, the correct accent for "私は思う" is "ワ➚タシワ　オ➚モ➘ウ", but it became "ワ➚タシ➘ワ　オ➚モ➘ウ" (due to the separate processing of "私" and "は").

Regarding the first bug, further applying `pyopenjtalk.g2p` to the reading "シャリョオ" of "車両" results in "シャリヨオ":

This behavior of `pyopenjtalk.g2p` (whether intentional or a bug) affected the subsequent accent processing, causing all accents after words like "車両" or "思う" to become "0".

(I am unsure whether this behavior of `pyopenjtalk.g2p` is intended or a bug)

# Changes in Model Structure

The main idea of "inputting not only the phoneme sequence but also semantic information obtained from BERT to enable content-aware reading" remains unchanged. However, there are some differences between the JP-Extra version and the existing 2.1-2.3 models, which will be discussed in this section.

## Basic Framework

Please refer to the [previous article](https://zenn.dev/litagin/articles/7179bb40f1f3a1).

To summarize, the voice generation (generator) part of (Style-)Bert-VITS2 consists of the following components:

- TextEncoder (receives text and returns various information)
- DurationPredictor (returns phoneme intervals, i.e., the duration of each phoneme)
- StochasticDurationPredictor (adds randomness to DurationPredictor?)
- Flow (contains voice tone information, especially pitch)
- Decoder (synthesizes the final voice output using various information; also contains voice tone information)

Furthermore, since the model uses a GAN for training, both the Generator (voice generation) and the Discriminator (distinguishes between real and generated voices) are trained. The following components correspond to the files saved during actual training:

- Generator: Has the above structure and generates voice from text. Saved in the `G_1000.pth` file. Only this is required for inference.
- MultiPeriodDiscriminator: I lack the knowledge to explain this. Saved in the `D_1000.pth` file. It is the main discriminator and is apparently used in HiFi-GAN and other models.
- DurationDiscriminator: Apparently a discriminator for phoneme intervals (output of DurationPredictor, etc). Saved in the `DUR_1000.pth` file.
- **WavLMDiscriminator**: Will be discussed in more detail later. Saved in the `WD_1000.pth` file. It seems to be a discriminator using the [WavLM](https://arxiv.org/abs/2110.13900) SSL model for voice.

The original JP-Extra version roughly follows the structure of Bert-VITS2 ver 2.3, with the addition of output control using CLAP prompts from ver 2.2.

The important structural points are as follows:

1. Use of WavLMDiscriminator introduced in ver 2.3
2. Removal of DurationDiscriminator
3. Increase of `gin_channels` parameter from 256 in 2.1 and 2.2 to 512, as in 2.3
4. Voice tone control using CLAP prompts

### 1. WavLMDiscriminator

I the knowledge and ability to provide a full explanation, so only the understood points will be discussed.

In the machine learning field, **SSL (Self-Supervised Learning) models**, which are trained on large amounts of unlabeled data, seem to be useful. These models are used as a base for downstream tasks.

In the voice field, the following models might be well-known:

- [HuBERT](https://arxiv.org/abs/2106.07447)
- [ContentVec](https://arxiv.org/abs/2204.09224)
- [wav2vec 2.0](https://arxiv.org/abs/2006.11477)
- [WavLM](https://arxiv.org/abs/2110.13900)

The **WavLM** model, specifically [microsoft/wavlm-base-plus](https://huggingface.co/microsoft/wavlm-base-plus), was introduced in Bert-VITS2 ver 2.3 and is also used in the JP-Extra version.

Although I do not fully understand the details, using the WavLM SSL model as a discriminator seems to improve the quality of the discriminator and, consequently, the quality of the generated voice. The `WD_*` files that started appearing in pre-trained models and during training in ver 2.3 and (Style-)Bert-VITS2 JP-Extra correspond to this model.

### 2. Removal of DurationDiscriminator

(Style-)Bert-VITS2 had been using DurationDiscriminator for some time, but since ver 2.3 (possibly due to compatibility with WavLMDiscriminator?), there have been issues with phoneme intervals becoming slightly unstable (sounding stretched or not stabilizing during training).

Considering this, the JP-Extra version does not use this component (so the `DUR_0` pre-trained model is no longer required).

As a result, you might get a slight impression of faster speech, but overall, the phoneme intervals seem to have settled down. The presence or absence of this DurationDiscriminator can be easily changed in the settings, so experimenting with it might yield some insights.

### 3. Addition of gin_channels

I can only provide impressions on this change, which was made in the original ver 2.3. There is a parameter called `gin_channels` that collectively determines the dimensions of the hidden layers in the model, and it was increased from 256 to 512.

In the previous comments on Bert-VITS2 2.3, I had the impression that increasing this value might have added a slight inability to fully train the model. However, considering the success of the JP-Extra version (which may have benefited from an increase in the amount of data used for pre-training), increasing the dimensions might have been a good change in the end (although this cannot be stated with certainty without experimenting with 256).

### 4. CLAP

This feature was introduced in the original Bert-VITS2 ver 2.2 (and removed in ver 2.3). It uses the [CLAP](https://huggingface.co/laion/clap-htsat-fused) model, which is trained on audio-text pairs, to attempt to control the output voice using text prompts (e.g., `Happy`).

Specifically, an embedding related to the CLAP model (which can extract feature vectors from both audio and text) is added to the TextEncoder part.

To be honest, while this may have been effective at the pre-trained model level, its effect seems to almost disappear when fine-tuning the model for practical use, and voice control using prompts is hardly possible, based on my experience.

# Conclusion

I encourage everyone to use Style-Bert-VITS2! It comes with tools for creating datasets and does not require Python environment setup for installation. As long as you have a GPU, you can easily train the model. Let's create our own tts models!