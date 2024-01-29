import os
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='0'
from transformers import WhisperFeatureExtractor
from transformers import WhisperProcessor
from transformers import WhisperTokenizer
from datasets import Audio ,load_dataset, DatasetDict, Dataset
import pandas as pd
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate
from transformers import WhisperForConditionalGeneration
from transformers import Seq2SeqTrainingArguments
from accelerate import Accelerator
from transformers import Seq2SeqTrainer





feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-large-v3", language="Arabic", task="transcribe")
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3", language="Arabic", task="transcribe")
test = pd.read_csv('/home/wakeb/Desktop/ASR_Dataset/Ahmed_Alghamdi/Yeman/data_yeman/test_yaman.csv',nrows = 100)
train = pd.read_csv('/home/wakeb/Desktop/ASR_Dataset/Ahmed_Alghamdi/Yeman/data_yeman/train_yaman.csv',nrows = 100)
#dataset = datasets.concatenate_datasets([test,train])
dataset = pd.concat([test,train]).dropna()
dataset = Dataset.from_pandas(dataset).remove_columns(['__index_level_0__'])
print(dataset)
dataset = dataset.cast_column("path", Audio(sampling_rate=16000))
dataset = dataset.train_test_split(0.1)
def prepare_dataset(batch):
    audio = batch["path"]
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["labels"] = tokenizer(batch["texts"]).input_ids
    return batch
dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names["train"], num_proc=16)
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch
data_collator = DataCollatorSpeechSeq2SeqWithPadding()
data_collator.processor = processor
metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-yemen-large/checkpoints",  
    per_device_train_batch_size=12,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=40000,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=1,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False
    #fsdp =True
    #push_to_hub=True,
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)
trainer.train()

