




from datasets import load_dataset, load_metric
import soundfile as sf

timit = load_dataset("timit_asr")

timit

"""Many ASR datasets only provide the target text, `'text'` for each audio file `'file'`. Timit actually provides much more information about each audio file, such as the `'phonetic_detail'`, etc., which is why many researchers choose to evaluate their models on phoneme classification instead of speech recognition when working with Timit. However, we want to keep the notebook as general as possible, so that we will only consider the transcribed text for fine-tuning.


"""

from datasets import ClassLabel
import random
import pandas as pd
from IPython.display import display, HTML

def show_random_elements(dataset, num_examples=10):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)

    df = pd.DataFrame(dataset[picks])
    display(HTML(df.to_html()))


show_random_elements(timit['train'].remove_columns(["file", "phonetic_detail", "word_detail", "dialect_region", "id", 
                                                    "sentence_type", "speaker_id"]), num_examples=20)

def get_phonetic_code(x):
    # print(x)
    x['phonetic_codes'] = \
        x['phonetic_detail']['utterance'][1:-1] # remove h#
    return x

timit = timit.map(get_phonetic_code)

print(timit['train']['text'][0])
print(timit['train']['phonetic_codes'][0])

# IPAに変換
# ref: https://en.wikipedia.org/wiki/ARPABET
arphabet_to_ipa = {
    'aa': 'ɑ',
    'ae': 'æ',
    'ah':'ʌ',
    'ao':'ɔ',
    'aw':'W',
    'ax':'ə',
    'axr':'ɚ',
    'ay':'Y',
    'eh':'ɛ',
    'er':'ɝ',
    'ey':'e',
    'ih':'ɪ',
    'ix':'ɨ',
    'iy':'i',
    'ow':'o',
    'oy':'O',
    'uh':'ʊ',
    'uw':'u',
    'ux':'ʉ',
    'b':'b',
    'ch':'C',
    'd':'d',
    'dh':'ð',
    'dx':'ɾ',
    'el':'l̩',
    'em':'m̩',
    'en':'n̩',
    'f':'f',
    'g':'g',
    'hh':'h',
    'h':'h',
    'jh':'J',
    'k':'k',
    'l':'l',    
    'm':'m',    
    'n':'n',    
    'ng':'ŋ',    
    'nx':'ɾ̃',    
    'p':'p',    
    'q':'ʔ',    
    'r':'ɹ',    
    's':'s',    
    'sh':'ʃ',    
    't':'t',    
    'th':'θ',    
    'v':'v',    
    'w':'w',    
    'wh':'ʍ',    
    'y':'j',    
    'z':'z',    
    'zh':'ʒ',    
    'ax-h':'ə̥',    
    'bcl':'b̚',    
    'dcl':'d̚',    
    'eng':'ŋ̍',    
    'gcl':'ɡ̚',    
    'hv':'ɦ',    
    'kcl':'k̚',    
    'pcl':'p̚',    
    'tcl':'t̚',
    'epi':'S', 
    'pau':'P',   
}

print(len(set(arphabet_to_ipa.keys())))
print(len(set(arphabet_to_ipa.values())))

# hhとhだけvalueがhで重複していることを確認
# from collections import Counter
# Counter(arphabet_to_ipa.values())

# Convert phonetic code to IPA represented in one character
def convert_to_ipa(x):
    # x['ipa'] = ' '.join([arphabet_to_ipa[code] for code in x['phonetic_codes']])
    x['ipa'] = ''.join([arphabet_to_ipa[code] for code in x['phonetic_codes']])
    return x

timit = timit.map(convert_to_ipa)

print(timit['train']['text'][0])
print(timit['train']['phonetic_codes'][0])
print(timit['train']['ipa'][0])

timit = timit.remove_columns(["text", "phonetic_detail", "word_detail", "dialect_region", "id", "sentence_type", "speaker_id"])

timit

"""Let's write a short function to display some random samples of the dataset and run it a couple of times to get a feeling for the transcriptions."""

from datasets import ClassLabel
import random
import pandas as pd
from IPython.display import display, HTML

def show_random_elements(dataset, num_examples=10):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)
    
    df = pd.DataFrame(dataset[picks])
    display(HTML(df.to_html()))

show_random_elements(timit["train"].remove_columns(["file"]), num_examples=20)

"""Alright! The transcriptions look very clean and the language seems to correspond more to written text than dialogue. This makes sense taking into account that [Timit](https://huggingface.co/datasets/timit_asr) is a read speech corpus.

We can see that the transcriptions contain some special characters, such as `,.?!;:`. Without a language model, it is much harder to classify speech chunks to such special characters because they don't really correspond to a characteristic sound unit. *E.g.*, the letter `"s"` has a more or less clear sound, whereas the special character `"."` does not.
Also in order to understand the meaning of a speech signal, it is usually not necessary to include special characters in the transcription.

In addition, we normalize the text to only have lower case letters and append a word separator token at the end.
"""

import re
# chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'
chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�]'

def remove_special_characters(batch):
    # batch["ipa"] = re.sub(chars_to_ignore_regex, '', batch["ipa"]).lower() + " "
    batch["ipa"] = re.sub(chars_to_ignore_regex, '', batch["ipa"]) + " "
    return batch

timit = timit.map(remove_special_characters)

show_random_elements(timit["train"].remove_columns(["file"]))

"""Good! This looks better. We have removed most special characters from transcriptions and normalized them to lower-case only.

In CTC, it is common to classify speech chunks into letters, so we will do the same here. 
Let's extract all distinct letters of the training and test data and build our vocabulary from this set of letters.

We write a mapping function that concatenates all transcriptions into one long transcription and then transforms the string into a set of chars. 
It is important to pass the argument `batched=True` to the `map(...)` function so that the mapping function has access to all transcriptions at once.
"""

def extract_all_chars(batch):
  all_text = " ".join(batch["ipa"])
  vocab = list(set(all_text))
  return {"vocab": [vocab], "all_text": [all_text]}

vocabs = timit.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=timit.column_names["train"])

"""Now, we create the union of all distinct letters in the training dataset and test dataset and convert the resulting list into an enumerated dictionary."""

vocab_list = list(set(vocabs["train"]["vocab"][0]) | set(vocabs["test"]["vocab"][0]))

vocab_dict = {v: k for k, v in enumerate(vocab_list)}
vocab_dict

"""Cool, we see that all letters of the alphabet occur in the dataset (which is not really surprising) and we also extracted the special characters `" "` and `'`. Note that we did not exclude those special characters because: 

- The model has to learn to predict when a word finished or else the model prediction would always be a sequence of chars which would make it impossible to separate words from each other.
- In English, we need to keep the `'` character to differentiate between words, *e.g.*, `"it's"` and `"its"` which have very different meanings.

To make it clearer that `" "` has its own token class, we give it a more visible character `|`. In addition, we also add an "unknown" token so that the model can later deal with characters not encountered in Timit's training set. 

Finally, we also add a padding token that corresponds to CTC's "*blank token*". The "blank token" is a core component of the CTC algorithm. For more information, please take a look at the "Alignment" section [here](https://distill.pub/2017/ctc/).
"""

vocab_dict["|"] = vocab_dict[" "]
del vocab_dict[" "]

vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)
len(vocab_dict)

"""Cool, now our vocabulary is complete and consists of 30 tokens, which means that the linear layer that we will add on top of the pretrained Wav2Vec2 checkpoint will have an output dimension of 30.

Let's now save the vocabulary as a json file.
"""

import json
with open('vocab.json', 'w') as vocab_file:
    json.dump(vocab_dict, vocab_file)

"""In a final step, we use the json file to instantiate an object of the `Wav2Vec2CTCTokenizer` class."""

from transformers import Wav2Vec2CTCTokenizer

tokenizer = Wav2Vec2CTCTokenizer("./vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")

"""Next, we will create the feature extractor.

### Create Wav2Vec2 Feature Extractor

Speech is a continuous signal and to be treated by computers, it first has to be discretized, which is usually called **sampling**. The sampling rate hereby plays an important role in that it defines how many data points of the speech signal are measured per second. Therefore, sampling with a higher sampling rate results in a better approximation of the *real* speech signal but also necessitates more values per second.

A pretrained checkpoint expects its input data to have been sampled more or less from the same distribution as the data it was trained on. The same speech signals sampled at two different rates have a very different distribution, *e.g.*, doubling the sampling rate results in data points being twice as long. Thus, 
before fine-tuning a pretrained checkpoint of an ASR model, it is crucial to verify that the sampling rate of the data that was used to pretrain the model matches the sampling rate of the dataset used to fine-tune the model.

Wav2Vec2 was pretrained on the audio data of [LibriSpeech](https://huggingface.co/datasets/librispeech_asr) and LibriVox which both were sampling with 16kHz. Our fine-tuning dataset, [Timit](hhtps://huggingface.co/datasets/timit_asr), was luckily also sampled with 16kHz. If the fine-tuning dataset would have been sampled with a rate lower or higher than 16kHz, we first would have had to up or downsample the speech signal to match the sampling rate of the data used for pretraining.

A Wav2Vec2 feature extractor object requires the following parameters to be instantiated:

- `feature_size`: Speech models take a sequence of feature vectors as an input. While the length of this sequence obviously varies, the feature size should not. In the case of Wav2Vec2, the feature size is 1 because the model was trained on the raw speech signal ${}^2$.
- `sampling_rate`: The sampling rate at which the model is trained on.
- `padding_value`: For batched inference, shorter inputs need to be padded with a specific value
- `do_normalize`: Whether the input should be *zero-mean-unit-variance* normalized or not. Usually, speech models perform better when normalizing the input
- `return_attention_mask`: Whether the model should make use of an `attention_mask` for batched inference. In general, models should **always** make use of the `attention_mask` to mask padded tokens. However, due to a very specific design choice of `Wav2Vec2`'s "base" checkpoint, better results are achieved when using no `attention_mask`. This is **not** recommended for other speech models. For more information, one can take a look at [this](https://github.com/pytorch/fairseq/issues/3227) issue. **Important** If you want to use this notebook to fine-tune [large-lv60](https://huggingface.co/facebook/wav2vec2-large-lv60), this parameter should be set to `True`.
"""

from transformers import Wav2Vec2FeatureExtractor

feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)

"""Great, Wav2Vec2's feature extraction pipeline is thereby fully defined!

To make the usage of Wav2Vec2 as user-friendly as possible, the feature extractor and tokenizer are *wrapped* into a single `Wav2Vec2Processor` class so that one only needs a `model` and `processor` object.
"""

from transformers import Wav2Vec2Processor

processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

"""If one wants to re-use the just created processor and the fine-tuned model of this notebook, one can mount his/her google drive to the notebook and save all relevant files there. To do so, please uncomment the following lines. 

We will give the fine-tuned model the name `"wav2vec2-base-timit-demo"`.
"""

# from google.colab import drive
# drive.mount('/content/gdrive/')

# processor.save_pretrained("/content/gdrive/MyDrive/wav2vec2-base-timit-demo")

"""Next, we can prepare the dataset.

### Preprocess Data

So far, we have not looked at the actual values of the speech signal but just kept the path to its file in the dataset. `Wav2Vec2` expects the audio file in the format of a 1-dimensional array, so in the first step, let's load all audio files into the dataset object.

Let's first check the serialization format of the downloaded audio files by looking at the first training sample.
"""

timit["test"][0]

"""Alright, the audio file is saved in the `.WAV` format. There are a couple of python-based libraries to read and process audio files, such as [librosa](https://github.com/librosa/librosa), [soundfile](https://github.com/bastibe/python-soundfile), and [audioread](https://github.com/beetbox/audioread). 

`librosa` seems to be the most active and prominent library, but since it depends on `soundfile` for loading of audio files, we will just use `soundfile` directly in this notebook.

An audio file usually stores both its values and the sampling rate with which the speech signal was digitalized. We want to store both in the dataset and write a `map(...)` function accordingly.
"""

import soundfile

def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = soundfile.read(batch["file"])
    batch["speech"] = speech_array
    batch["sampling_rate"] = sampling_rate
    batch["target_text"] = batch["ipa"]
    return batch

timit = timit.map(speech_file_to_array_fn, remove_columns=timit.column_names["train"], num_proc=4)

"""Great, let's listen to a couple of audio files to better understand the dataset and verify that the audio was correctly loaded. 

**Note**: *You can click the following cell a couple of times to listen to different speech samples.*
"""

import IPython.display as ipd
import numpy as np
import random

rand_int = random.randint(0, len(timit["train"]))

ipd.Audio(data=np.asarray(timit["train"][rand_int]["speech"]), autoplay=True, rate=16000)

"""It can be heard, that the speakers change along with their speaking rate, accent, etc. Overall, the recordings sound relatively clear though, which is to be expected from a read speech corpus.

Let's do a final check that the data is correctly prepared, but printing the shape of the speech input, its transcription, and the corresponding sampling rate.

**Note**: *You can click the following cell a couple of times to verify multiple samples.*
"""

rand_int = random.randint(0, len(timit["train"]))

print("Target text:", timit["train"][rand_int]["target_text"])
print("Input array shape:", np.asarray(timit["train"][rand_int]["speech"]).shape)
print("Sampling rate:", timit["train"][rand_int]["sampling_rate"])

"""Good! Everything looks fine - the data is a 1-dimensional array, the sampling rate always corresponds to 16kHz, and the target text is normalized.

Finally, we can process the dataset to the format expected by the model for training. We will again make use of the `map(...)` function.

First, we check that all data samples have the same sampling rate (of 16kHz).
Second, we extract the `input_values` from the loaded audio file. In our case, this includes only normalization, but for other speech models, this step could correspond to extracting, *e.g.* [Log-Mel features](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum). 
Third, we encode the transcriptions to label ids.

**Note**: This mapping function is a good example of how the `Wav2Vec2Processor` class should be used. In "normal" context, calling `processor(...)` is redirected to `Wav2Vec2FeatureExtractor`'s call method. When wrapping the processor into the `as_target_processor` context, however, the same method is redirected to `Wav2Vec2CTCTokenizer`'s call method.
For more information please check the [docs](https://huggingface.co/transformers/master/model_doc/wav2vec2.html#transformers.Wav2Vec2Processor.__call__).
"""

def prepare_dataset(batch):
    # check that all files have the correct sampling rate
    assert (
        len(set(batch["sampling_rate"])) == 1
    ), f"Make sure all inputs have the same sampling rate of {processor.feature_extractor.sampling_rate}."

    batch["input_values"] = processor(batch["speech"], sampling_rate=batch["sampling_rate"][0]).input_values
    
    with processor.as_target_processor():
        batch["labels"] = processor(batch["target_text"]).input_ids
    return batch

timit_prepared = timit.map(prepare_dataset, remove_columns=timit.column_names["train"], batch_size=8, num_proc=4, batched=True)

"""## Training & Evaluation

The data is processed so that we are ready to start setting up the training pipeline. We will make use of 🤗's [Trainer](https://huggingface.co/transformers/master/main_classes/trainer.html?highlight=trainer) for which we essentially need to do the following:

- Define a data collator. In contrast to most NLP models, Wav2Vec2 has a much larger input length than output length. *E.g.*, a sample of input length 50000 has an output length of no more than 100. Given the large input sizes, it is much more efficient to pad the training batches dynamically meaning that all training samples should only be padded to the longest sample in their batch and not the overall longest sample. Therefore, fine-tuning Wav2Vec2 requires a special padding data collator, which we will define below

- Evaluation metric. During training, the model should be evaluated on the word error rate. We should define a `compute_metrics` function accordingly

- Load a pretrained checkpoint. We need to load a pretrained checkpoint and configure it correctly for training.

- Define the training configuration.

After having fine-tuned the model, we will correctly evaluate it on the test data and verify that it has indeed learned to correctly transcribe speech.

### Set-up Trainer

Let's start by defining the data collator. The code for the data collator was copied from [this example](https://github.com/huggingface/transformers/blob/9a06b6b11bdfc42eea08fa91d0c737d1863c99e3/examples/research_projects/wav2vec2/run_asr.py#L81).

Without going into too many details, in contrast to the common data collators, this data collator treats the `input_values` and `labels` differently and thus applies to separate padding functions on them (again making use of Wav2Vec2's context manager). This is necessary because in speech input and output are of different modalities meaning that they should not be treated by the same padding function.
Analogous to the common data collators, the padding tokens in the labels with `-100` so that those tokens are **not** taken into account when computing the loss.
"""

import torch

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

"""Next, the evaluation metric is defined. As mentioned earlier, the 
predominant metric in ASR is the word error rate (WER), hence we will use it in this notebook as well.
"""

# wer_metric = load_metric("wer")
# cer_metric = load_metric('ctl/wav2vec2-large-xlsr-cantonese/cer.py')

"""The model will return a sequence of logit vectors:
$\mathbf{y}_1, \ldots, \mathbf{y}_m$ with $\mathbf{y}_1 = f_{\theta}(x_1, \ldots, x_n)[0]$ and $n >> m$.

A logit vector $\mathbf{y}_1$ contains the log-odds for each word in the vocabulary we defined earlier, thus $\text{len}(\mathbf{y}_i) =$ `config.vocab_size`. We are interested in the most likely prediction of the model and thus take the `argmax(...)` of the logits. Also, we transform the encoded labels back to the original string by replacing `-100` with the `pad_token_id` and decoding the ids while making sure that consecutive tokens are **not** grouped to the same token in CTC style ${}^1$.
"""

from jiwer import wer

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    # wer = wer_metric.compute(predictions=pred_str, references=label_str)
    # cer = fastwer.score(pred_str, label_str, char_level=True)
    # cer = word_error_rate(hypotheses=pred_str, references=label_str, use_cer=True)

    # ref: https://huggingface.co/ctl/wav2vec2-large-xlsr-cantonese/blob/main/cer.py
    preds = [char for seq in pred_str for char in list(seq)]
    refs = [char for seq in label_str for char in list(seq)]
    cer = wer(refs, preds)

    return {"cer": cer}
    # return {"wer": wer}

# testing
label_str = 'aab'
pred_str = 'aaac'
preds = [char for seq in pred_str for char in list(seq)]
refs = [char for seq in label_str for char in list(seq)]
cer = wer(refs, preds)
print(cer)

"""Now, we can load the pretrained `Wav2Vec2` checkpoint. The tokenizer's `pad_token_id` must be to define the model's `pad_token_id` or in the case of `Wav2Vec2ForCTC` also CTC's *blank token* ${}^2$. To save GPU memory, we enable PyTorch's [gradient checkpointing](https://pytorch.org/docs/stable/checkpoint.html) and also set the loss reduction to "*mean*"."""

from transformers import Wav2Vec2ForCTC

# model = Wav2Vec2ForCTC.from_pretrained(
#     "facebook/wav2vec2-base", 
#     gradient_checkpointing=True, 
#     ctc_loss_reduction="mean", 
#     pad_token_id=processor.tokenizer.pad_token_id,
# )

model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-large-xlsr-53", 
    attention_dropout=0.1,
    hidden_dropout=0.1,
    feat_proj_dropout=0.0,
    mask_time_prob=0.05,
    layerdrop=0.1,
    gradient_checkpointing=True, 
    ctc_loss_reduction="mean", 
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer)
)

"""The first component of Wav2Vec2 consists of a stack of CNN layers that are used to extract acoustically meaningful - but contextually independent - features from the raw speech signal. This part of the model has already been sufficiently trained during pretrainind and as stated in the [paper](https://arxiv.org/abs/2006.11477) does not need to be fine-tuned anymore. 
Thus, we can set the `requires_grad` to `False` for all parameters of the *feature extraction* part.
"""

model.freeze_feature_extractor()

"""In a final step, we define all parameters related to training. 
To give more explanation on some of the parameters:
- `group_by_length` makes training more efficient by grouping training samples of similar input length into one batch. This can significantly speed up training time by heavily reducing the overall number of useless padding tokens that are passed through the model
- `learning_rate` and `weight_decay` were heuristically tuned until fine-tuning has become stable. Note that those parameters strongly depend on the Timit dataset and might be suboptimal for other speech datasets.

For more explanations on other parameters, one can take a look at the [docs](https://huggingface.co/transformers/master/main_classes/trainer.html?highlight=trainer#trainingarguments).

**Note**: If one wants to save the trained models in his/her google drive the commented-out `output_dir` can be used instead.
"""

# from transformers import TrainingArguments

# training_args = TrainingArguments(
#   # output_dir="/content/gdrive/MyDrive/wav2vec2-base-timit-demo",
#   output_dir="./wav2vec2-base-timit-demo",
#   group_by_length=True,
#   per_device_train_batch_size=32,
#   evaluation_strategy="steps",
#   num_train_epochs=30,
#   fp16=True,
#   save_steps=500,
#   eval_steps=500,
#   logging_steps=500,
#   learning_rate=1e-4,
#   weight_decay=0.005,
#   warmup_steps=1000,
#   save_total_limit=2,
# )

from transformers import TrainingArguments

training_args = TrainingArguments(
  # output_dir=model_dir,
  output_dir="./wav2vec2-base-timit-demo",
  # output_dir="./wav2vec2-large-xlsr-turkish-demo",
  group_by_length=True,
  per_device_train_batch_size=16,
#   per_device_train_batch_size=32,
  gradient_accumulation_steps=2,
  evaluation_strategy="steps",
  num_train_epochs=10,
  fp16=True,
  save_steps=100,
  eval_steps=100,
  logging_steps=100,
  learning_rate=3e-4,
  warmup_steps=500,
  save_total_limit=2,
)

"""Now, all instances can be passed to Trainer and we are ready to start training!"""

from transformers import Trainer

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=timit_prepared["train"],
    eval_dataset=timit_prepared["test"],
    tokenizer=processor.feature_extractor,
)

"""---

${}^1$ To allow models to become independent of the speaker rate, in CTC, consecutive tokens that are identical are simply grouped as a single token. However, the encoded labels should not be grouped when decoding since they don't correspond to the predicted tokens of the model, which is why the `group_tokens=False` parameter has to be passed. If we wouldn't pass this parameter a word like `"hello"` would incorrectly be encoded, and decoded as `"helo"`.

${}^2$ The blank token allows the model to predict a word, such as `"hello"` by forcing it to insert the blank token between the two l's. A CTC-conform prediction of `"hello"` of our model would be `[PAD] [PAD] "h" "e" "e" "l" "l" [PAD] "l" "o" "o" [PAD]`.

### Training

Training will take between 90 and 180 minutes depending on the GPU allocated to this notebook. While the trained model yields satisfying results on *Timit*'s test data, it is by no means an optimally fine-tuned model. The purpose of this notebook is to demonstrate how Wav2Vec2's [base](https://huggingface.co/facebook/wav2vec2-base), [large](https://huggingface.co/facebook/wav2vec2-large), and [large-lv60](https://huggingface.co/facebook/wav2vec2-large-lv60) checkpoints can be fine-tuned on any English dataset.

In case you want to use this google colab to fine-tune your model, you should make sure that your training doesn't stop due to inactivity. A simple hack to prevent this is to paste the following code into the console of this tab (*right mouse click -> inspect -> Console tab and insert code*).

```javascript
function ConnectButton(){
    console.log("Connect pushed"); 
    document.querySelector("#top-toolbar > colab-connect-button").shadowRoot.querySelector("#connect").click() 
}
setInterval(ConnectButton,60000);
```
"""

trainer.train(resume_from_checkpoint=True)

"""The final WER should be below 0.3 which is reasonable given that state-of-the-art phoneme error rates (PER) are just below 0.1 (see [leaderboard](https://paperswithcode.com/sota/speech-recognition-on-timit)) and that WER is usually worse than PER.

The resulting model of this notebook has been saved to [patrickvonplaten/wav2vec2-base-timit-demo](https://huggingface.co/patrickvonplaten/wav2vec2-base-timit-demo).

### Evaluate

In the final part, we evaluate our fine-tuned model on the test set and play around with it a bit.

Let's load the `processor` and `model`.
"""

path = 'wav2vec2-base-timit-demo/checkpoint-500'
finetuned_model = Wav2Vec2ForCTC.from_pretrained(path)

# processor = Wav2Vec2Processor.from_pretrained("patrickvonplaten/wav2vec2-base-timit-demo")

# model = Wav2Vec2ForCTC.from_pretrained("patrickvonplaten/wav2vec2-base-timit-demo")

"""Now, we will make use of the `map(...)` function to predict the transcription of every test sample and to save the prediction in the dataset itself. We will call the resulting dictionary `"results"`. 

**Note**: we evaluate the test data set with `batch_size=1` on purpose due to this [issue](https://github.com/pytorch/fairseq/issues/3227). Since padded inputs don't yield the exact same output as non-padded inputs, a better WER can be achieved by not padding the input at all.
"""

def map_to_result(batch):
  finetuned_model.to("cuda")
  input_values = processor(
      batch["speech"], 
      sampling_rate=batch["sampling_rate"], 
      return_tensors="pt"
  ).input_values.to("cuda")

  with torch.no_grad():
    logits = finetuned_model(input_values).logits

  pred_ids = torch.argmax(logits, dim=-1)
  batch["pred_str"] = processor.batch_decode(pred_ids)[0]
  
  return batch

results = timit["test"].map(map_to_result)

"""Let's compute the overall WER now."""

print("Test WER: {:.3f}".format(wer_metric.compute(predictions=results["pred_str"], references=results["target_text"])))

"""18.6% WER - not bad! Our model would have surely made the top 20 on the official [leaderboard](https://paperswithcode.com/sota/speech-recognition-on-timit).

Let's take a look at some predictions to see what errors are made by the model.
"""

show_random_elements(results.remove_columns(["speech", "sampling_rate"]))

"""It becomes clear that the predicted transcriptions are acoustically very similar to the target transcriptions, but often contain spelling or grammatical errors. This shouldn't be very surprising though given that we purely rely on Wav2Vec2 without making use of a language model.

Finally, to better understand how CTC works, it is worth taking a deeper look at the exact output of the model. Let's run the first test sample through the model, take the predicted ids and convert them to their corresponding tokens.
"""

model.to("cuda")
input_values = processor(timit["test"][0]["speech"], sampling_rate=timit["test"][0]["sampling_rate"], return_tensors="pt").input_values.to("cuda")

with torch.no_grad():
  logits = model(input_values).logits

pred_ids = torch.argmax(logits, dim=-1)

# convert ids to tokens
" ".join(processor.tokenizer.convert_ids_to_tokens(pred_ids[0].tolist()))

"""The output should make it a bit clearer how CTC works in practice. The model is to some extent invariant to speaking rate since it has learned to either just repeat the same token in case the speech chunk to be classified still corresponds to the same token. This makes CTC a very powerful algorithm for speech recognition since the speech file's transcription is often very much independent of its length.

I again advise the reader to take a look at [this](https://distill.pub/2017/ctc) very nice blog post to better understand CTC.
"""
