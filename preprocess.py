# -*- coding: utf-8 -*-


import transformers
import torch

import json
import logging
import os
import random
from pathlib import Path

from transformers import DistilBertTokenizerFast

class SquadPreprocessor:
    def __init__(self):
        #self.folder = folder
        self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    @staticmethod
    def _read_squad(path: str, frac=1.0, include_impossible=False) -> (list, list, list):
     

        path = Path(path)
        with open(path, 'rb') as f:
            squad_dict = json.load(f)

        contexts = []
        questions = []
        answers = []
        is_impossible = []
        logging.info(f"Including plausible answers: {include_impossible}")
        for group in squad_dict['data']:
            for passage in group['paragraphs']:
                if random.random() > frac:  # Skip if random sampling is enabled
                    continue
                context = passage['context']
                for qa in passage['qas']:
                    question = qa['question']

                    ans = qa['answers']
                    impossible = 0

                    if include_impossible:  # For model with impossible answers
                        if qa['is_impossible']:
                            ans = qa['plausible_answers']
                            impossible = 1

                    for answer in ans:
                        contexts.append(context)
                        questions.append(question)
                        answers.append(answer)
                        is_impossible.append(impossible)

        return contexts, questions, answers, is_impossible

    @staticmethod
    def _add_end_idx(answers: list, contexts: list) -> None:
        

        for answer, context in zip(answers, contexts):
            gold_text = answer['text']
            start_idx = answer['answer_start']
            end_idx = start_idx + len(gold_text)

            # sometimes squad answers are off by a character or two – fix this
            if context[start_idx:end_idx] == gold_text:
                answer['answer_end'] = end_idx
            elif context[start_idx - 1:end_idx - 1] == gold_text:
                answer['answer_start'] = start_idx - 1
                answer['answer_end'] = end_idx - 1  # When the gold label is off by one character
            elif context[start_idx - 2:end_idx - 2] == gold_text:
                answer['answer_start'] = start_idx - 2
                answer['answer_end'] = end_idx - 2  # When the gold label is off by two characters

    def _add_token_positions(self, encodings, answers):
        start_positions = []
        end_positions = []
        for i in range(len(answers)):
            start_positions.append(encodings.char_to_token(i, answers[i]['answer_start']))
            end_positions.append(encodings.char_to_token(i, answers[i]['answer_end'] - 1))
            # if None, the answer passage has been truncated
            if start_positions[-1] is None:
                start_positions[-1] = self.tokenizer.model_max_length
            if end_positions[-1] is None:
                end_positions[-1] = self.tokenizer.model_max_length
        encodings.update({'start_positions': start_positions, 'end_positions': end_positions})

    #   encodings.update({'is_impossible': is_impossible})

    def get_encodings(self, random_sample_train: float = 0.1, random_sample_val: float = 0.1, include_impossible=False,
                      **tokenizer_kwargs):
            

        train_contexts, train_questions, train_answers, is_impossible_train = self._read_squad(
            path="/content/drive/MyDrive/train-v2.0.json",
            frac=random_sample_train, include_impossible=include_impossible)
        val_contexts, val_questions, val_answers, is_impossible_val = self._read_squad(
            path="/content/drive/MyDrive/dev-v2.0.json",
            frac=random_sample_val,
            include_impossible=include_impossible)

        

        self._add_end_idx(train_answers, train_contexts)
        self._add_end_idx(val_answers, val_contexts)

        train_encodings = self.tokenizer(train_contexts, train_questions, truncation=True, padding=True,
                                         **tokenizer_kwargs)
        val_encodings = self.tokenizer(val_contexts, val_questions, truncation=True, padding=True, **tokenizer_kwargs)

        logging.info(
            f"Number of impossible questions, train: {sum(is_impossible_train)}, val: {sum(is_impossible_val)}")

        self._add_token_positions(train_encodings, train_answers)
        self._add_token_positions(val_encodings, val_answers)
        return train_encodings, val_encodings

class SquadPlausibleAnswersPreprocessor(SquadPreprocessor):
    """
    Preprocess that includes plausible answers
    """

    def __init__(self):
        super().__init__()

    def get_encodings(self, random_sample_train: float = 0.1, random_sample_val: float = 0.1,
                      **tokenizer_kwargs):
        return super().get_encodings(random_sample_train=random_sample_train, random_sample_val=random_sample_val,
                                     include_impossible=True, **tokenizer_kwargs)

from model import QAModel

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    from transformers import DistilBertModel

    sp = SquadPreprocessor()
    sp = SquadPlausibleAnswersPreprocessor()
    train_enc, val_enc = sp.get_encodings(random_sample_train=0.001, random_sample_val=0.1, return_tensors="pt")

    # Decoding
    #    print(sp.tokenizer.decode(train_enc['input_ids'][0]))
    dbm = DistilBertModel.from_pretrained('distilbert-base-uncased', return_dict=True)
    model = QAModel(transformer_model=dbm, device= torch.device("cpu"))

    #model = DistilBertModel.from_pretrained('distilbert-base-uncased', return_dict=True)
    #out = model(**train_enc)

    print("End")

from google.colab import drive
drive.mount('/content/drive')
