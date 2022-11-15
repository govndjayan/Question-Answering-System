


import logging
import os

import torch
import transformers
from transformers import DistilBertModel, DistilBertTokenizerFast

from model import QAModel

logging.basicConfig(level=logging.INFO)

class QAModelInference:
    """
    This class combines output of models trained on possible only question and possible+impossible questions
    to produce textual output and corresponding probabilities.
    """

    def __init__(self, models_path, plausible_model_fn, possible_model_fn, device="cpu"):
        self.plausible_model_fn = plausible_model_fn
        self.possible_model_fn = possible_model_fn
        self.models_path = models_path
        self.device = device

        models = self._check_load_models()
        self.possible_model = models[0]
        self.plausible_model = models[1]

        self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    def _check_load_models(self) -> list:
        """
        Checks if models exist and loads into memory.
        Returns
        -------
        list [possible_model, plausble_model], containing initialized instances of PyTorch models.
        """

        models = []
        for model_name in (self.possible_model_fn, self.plausible_model_fn):
            fn = os.path.join(self.models_path, model_name)
            logging.info(f"Loading {fn}")
            if not os.path.exists(fn):
                raise FileExistsError(f"Model {fn} doesn't exist. Please run training first.")
            models.append(self.load_model(fn))
        return models

    def load_model(self, state_path):
        """
        Initialises the model and loads saved state into the instance of the model.
        Parameters
        ----------
        state_path (str) - path pointing to the saved state.
        Returns
        -------
        Model (torch.nn.Module)
        """

        logging.info(f"Loading trained state from {state_path}")
        dbm = DistilBertModel.from_pretrained('distilbert-base-uncased', return_dict=True)
        device = torch.device(self.device)
        dbm.to(device)
        model = QAModel(transformer_model=dbm, device=device)

        # checkpoint = torch.load(state_path, map_location=device)
        model.load_state_dict(torch.load(state_path), strict = False)
        model.eval()  # Switch to evaluation mode

        return model

    def get_model_data(self, model, context: str, question: str):
       
        tokens = self.tokenizer(context, question, truncation=True, padding=True, return_tensors="pt")
        start_logit, _, end_logit, _ = model(tokens)

        # Convert to proper probabilities
        start_logit, end_logit = torch.softmax(start_logit, dim=1), torch.softmax(end_logit, dim=1)
        start_idx, end_idx = torch.argmax(start_logit), torch.argmax(end_logit) + 1

        words = ""
        if end_idx < start_idx:
            end_idx = torch.argmax(end_logit[0][start_idx:]) + 1
            logging.warning(f"Error: start_idx = {start_idx}, end_idx = {end_idx}")
        else:
            input_ids = tokens['input_ids'].squeeze(0)
            words = self.tokenizer.decode(token_ids=input_ids[start_idx:end_idx].to('cpu').numpy())

        return start_idx, end_idx, words, start_logit.detach().to('cpu').numpy(), end_logit.detach().to('cpu').numpy()

    def extract_answer(self, context: str, question: str):

        # Get data for possible answers
        start_po, end_po, words_po, start_proba_po, end_proba_po = self.get_model_data(self.possible_model, context,
                                                                                       question)

        start_pl, end_pl, words_pl, start_proba_pl, end_proba_pl = self.get_model_data(self.plausible_model, context,
                                                                                       question)

        if start_po != start_pl and end_po != end_pl:
            ans = self._form_answer("<ANSWER UNKNOWN>", '', start_proba_po, end_proba_po, start_proba_pl,
                                    end_proba_pl,
                                    start_po, end_po, start_pl, end_pl)

            # As a plausible answer, return one with highest probability
            if max(start_proba_po[0]) + max(end_proba_po[0]) > max(start_proba_pl[0]) + max(end_proba_pl[0]):
                ans['plausible_answer'] = words_po
            else:
                ans['plausible_answer'] = words_pl
            return ans

        return self._form_answer(words_po, '', start_proba_po, end_proba_po, start_proba_pl, end_proba_pl,
                                 start_po, end_po,
                                 start_pl, end_pl)

    def _form_answer(self, answer_possible, answer_plausible, start_proba_po, end_proba_po, start_proba_pl, end_proba_pl,
                     start_po, end_po, start_pl, end_pl):
        """
        Forms an output dictionary.
        """

        return {
            'answer': answer_possible,
            'plausible_answer': answer_plausible,
            'start_word_proba_possible_model': start_proba_po,
            'end_word_proba_possible_model': end_proba_po,
            'start_word_proba_plausible_model': start_proba_pl,
            'end_word_proba_plausible_model': end_proba_pl,
            'start_position_possible_model': start_po,
            'end_position_possible_model': end_po,
            'start_position_plausible_model': start_pl,
            'end_position_plausible_model': end_pl,

        }


def save_lean_model(model, path):
    torch.save(model.state_dict(), path, _use_new_zipfile_serialization=False)


def load_model(state_path, device="cpu"):
    logging.info(f"Loading trained state from {state_path}")
    dbm = DistilBertModel.from_pretrained('distilbert-base-uncased', return_dict=True)
    device = torch.device(device)
    dbm.to(device)
    model = QAModel(transformer_model=dbm, device=device)

    checkpoint = torch.load(state_path, map_location=device)
    model=model.load_state_dict(dict([(n, p) for n, p in checkpoint['model_state_dict'].items()]), strict=False)
    model.eval()  # Switch to evaluation mode

    return model

if __name__ == '__main__':
    pass
