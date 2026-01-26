from dataclasses import dataclass
from typing import List, Union, Optional
from transformers import Pipeline
import torch
import numpy as np
from .rep_readers import DIRECTION_FINDERS, RepReader
from .rep_reading_pipeline import RepReadingPipeline

class RepReadingNProbCalcPipeline(RepReadingPipeline):

    def __init__(self, user_tag, assistant_tag, **kwargs):
        super().__init__(**kwargs)
        self.user_tag = user_tag
        self.assistant_tag = assistant_tag
 
    def _get_hidden_states(
            self, 
            outputs,
            rep_token: Union[str, int, list]=-1,
            hidden_layers: Union[List[int], int]=-1,
            which_hidden_states: Optional[str]=None):
        
        if hasattr(outputs, 'encoder_hidden_states') and hasattr(outputs, 'decoder_hidden_states'):
            outputs['hidden_states'] = outputs[f'{which_hidden_states}_hidden_states']
    
        hidden_states_layers = {}
        for layer in hidden_layers:
            hidden_states = outputs['hidden_states'][layer]
            hidden_states =  hidden_states[:, rep_token, :]
            # hidden_states_layers[layer] = hidden_states.cpu().to(dtype=torch.float32).detach().numpy()
            hidden_states_layers[layer] = hidden_states.detach()

        return hidden_states_layers

    def _sanitize_parameters(self, 
                             rep_reader: RepReader=None,
                             rep_token: Union[str, int]=-1,
                             hidden_layers: Union[List[int], int]=-1,
                             component_index: int=0,
                             which_hidden_states: Optional[str]=None,
                             **tokenizer_kwargs):
        preprocess_params = tokenizer_kwargs
        forward_params =  {}
        postprocess_params = {}

        forward_params['rep_token'] = rep_token

        if not isinstance(hidden_layers, list):
            hidden_layers = [hidden_layers]


        assert rep_reader is None or len(rep_reader.directions) == len(hidden_layers), f"expect total rep_reader directions ({len(rep_reader.directions)})== total hidden_layers ({len(hidden_layers)})"                 
        forward_params['rep_reader'] = rep_reader
        forward_params['hidden_layers'] = hidden_layers
        forward_params['component_index'] = component_index
        forward_params['which_hidden_states'] = which_hidden_states
        
        return preprocess_params, forward_params, postprocess_params
   
    def preprocess(
            self, 
            inputs: Union[tuple, str],
            **tokenizer_kwargs):

        if self.image_processor:
            return self.image_processor(inputs, add_end_of_utterance_token=False, return_tensors="pt")
        
        return self.tokenizer(inputs, return_tensors=self.framework, **tokenizer_kwargs)
        # if isinstance(inputs, str): # to get repre_reader
        #     return self.tokenizer(inputs, return_tensors=self.framework, **tokenizer_kwargs)
        
        # assert type(inputs) == tuple, f"inputs must be a tuple or string, but got {type(inputs)}"
        # question, answer = inputs
        # ttl_text = f"{question.strip()}: {answer.strip()}" 
        
        # return  self.tokenizer(ttl_text, return_tensors=self.framework, **tokenizer_kwargs)


    def postprocess(self, outputs):
        if 'logits' not in outputs: return outputs
        assert outputs['input_ids'].shape[0] == 1, "Batch size must be 1"
        
        if all(sp_tk in self.tokenizer.special_tokens_map['additional_special_tokens'] for sp_tk in [self.user_tag, self.assistant_tag]):
            assistant_tag_pos = np.where(outputs['input_ids'].numpy()==self.tokenizer.encode(self.assistant_tag,add_special_tokens=False)[0]) # (array([0,1]), array([39,38]))
            # check that every sample in the batch has and only has one assistant_tag_pos
            assert np.unique(assistant_tag_pos[0]) == assistant_tag_pos[0], f"Batch must have only one assistant tag,  assistant_tag_pos {assistant_tag_pos} "
            first_diff_positions = [ pos+1 for pos in assistant_tag_pos[1].tolist()]
        else:
            meaningfull_input_ids = outputs['input_ids'].flatten()[-outputs['attention_mask'].sum():]
            raw_texts = self.tokenizer.decode(meaningfull_input_ids, skip_special_tokens=True)
            query, ans = raw_texts.split(self.assistant_tag,1)
            query_input_ids = self.tokenizer(query, return_tensors="pt")['input_ids']
            assert torch.allclose(query_input_ids[0][:4], meaningfull_input_ids[:4]), "Prompt does not match at the beginning"
        
            first_diff_positions = self.find_first_difference(query_input_ids, outputs['input_ids'])
            
        ans_probabilities, ans_ids = self.calculate_sentence_probability(outputs['logits'], outputs['input_ids'], first_diff_positions)
        outputs['ans_probabilities'] = ans_probabilities
        outputs['ans_ids'] = ans_ids
        
        return outputs

    def find_first_difference(self, input_ids, full_input_ids):
        batch_size = input_ids.size(0)
        assert batch_size == full_input_ids.size(0), "Batch sizes must match"
        first_diff_positions = []
        
        for batch_id in range(batch_size):
            input_ids_flat = input_ids[batch_id].flatten()
            full_input_ids_flat = full_input_ids[batch_id].flatten()
            
            # Consider the paddings
            input_ids_mask = input_ids_flat != self.tokenizer.pad_token_id
            full_input_ids_mask = full_input_ids_flat != self.tokenizer.pad_token_id
            
            input_ids_flat = input_ids_flat[input_ids_mask]
            full_input_ids_flat = full_input_ids_flat[full_input_ids_mask]
            
            min_length = min(len(input_ids_flat), len(full_input_ids_flat))
            
            for i in range(min_length):
                if input_ids_flat[i] != full_input_ids_flat[i]:
                    first_diff_positions.append(i + (full_input_ids_mask==0).sum().item())
                    break
            else:
                first_diff_positions.append(len(input_ids_flat))  # No difference found

        assert len(first_diff_positions) == batch_size, "Must find a position for each item in batch"
        return first_diff_positions   
    
    def calculate_sentence_probability(self,logits, full_input_ids, start_positions):
        batch_probabilities, batch_ans_ids = [],[]
        for batch_idx in range(len(logits)):
            probabilities = []
            start_pos = start_positions[batch_idx]
            for i in range(start_pos, full_input_ids.shape[1]-1):
                next_token_logits = logits[batch_idx, i, :]
                next_token_probs = torch.softmax(next_token_logits, dim=0)
                target_token_id = full_input_ids[batch_idx, i + 1]
                if target_token_id == self.tokenizer.pad_token_id:
                    break  # Stop at padding
                target_token_prob = next_token_probs[target_token_id].item()
                probabilities.append(target_token_prob)
    
            assert len(probabilities) > 0, f"No probabilities calculated for prompt {batch_idx}"
            non_zero_probs = [p for p in probabilities if p != 0]
            sentence_prob = torch.prod(torch.tensor(non_zero_probs)).item()
            ans_ids = full_input_ids[batch_idx, start_pos:]
            batch_probabilities.append((sentence_prob, probabilities))
            batch_ans_ids.append(ans_ids)
            
        assert len(batch_probabilities) == len(logits), "Must have results for each prompt"
        return batch_probabilities, batch_ans_ids

    def _forward(self, model_inputs,  hidden_layers, rep_token=None,rep_reader=None, component_index=0, which_hidden_states=None, pad_token_id=None):
        """
        Args:
        - which_hidden_states (str): Specifies which part of the model (encoder, decoder, or both) to compute the hidden states from. 
                        It's applicable only for encoder-decoder models. Valid values: 'encoder', 'decoder'.
        Return: 
        - hidden_states (dict): A dictionary with keys as layer numbers and values as rep_token's projection at PCA direction
        """
        if rep_token is None:
            rep_token = list(range(model_inputs['input_ids'].size(1)))
        
        # get model hidden states and optionally transform them with a RepReader
        with torch.no_grad():
            if hasattr(self.model, "encoder") and hasattr(self.model, "decoder"):
                decoder_start_token = [self.tokenizer.pad_token] * model_inputs['input_ids'].size(0)
                decoder_input = self.tokenizer(decoder_start_token, return_tensors="pt").input_ids
                model_inputs['decoder_input_ids'] = decoder_input
            outputs =  self.model(**model_inputs, output_hidden_states=True)
        hidden_states = self._get_hidden_states(outputs, rep_token, hidden_layers, which_hidden_states)
        
        if rep_reader is None:
            return hidden_states
        
        layer2rep_trans = rep_reader.transform(hidden_states, hidden_layers, component_index)
        return_dict = {
            'logits': outputs.logits,
            'input_ids': model_inputs['input_ids'],
            'attention_mask': model_inputs['attention_mask'],
        }
        return_dict.update(layer2rep_trans)
        
        return return_dict
        


    def _batched_string_to_hiddens(self, train_inputs, rep_token, hidden_layers, batch_size, which_hidden_states, **tokenizer_args):
        # Wrapper method to get a dictionary hidden states from a list of strings
        hidden_states_outputs = self(train_inputs, rep_token=rep_token,
            hidden_layers=hidden_layers, batch_size=batch_size, rep_reader=None, which_hidden_states=which_hidden_states, **tokenizer_args)
        hidden_states = {layer: [] for layer in hidden_layers}
        for hidden_states_batch in hidden_states_outputs:
            for layer in hidden_states_batch:
                hidden_states[layer].extend(hidden_states_batch[layer])
        return {k: np.vstack(v) for k, v in hidden_states.items()}
    
    def _validate_params(self, n_difference, direction_method):
        # validate params for get_directions
        if direction_method == 'clustermean':
            assert n_difference == 1, "n_difference must be 1 for clustermean"

    def get_directions(
            self, 
            train_inputs: Union[str, List[str], List[List[str]]], 
            rep_token: Union[str, int]=-1, 
            hidden_layers: Union[str, int]=-1,
            n_difference: int = 1,
            batch_size: int = 8, 
            train_labels: List[int] = None,
            direction_method: str = 'pca',
            direction_finder_kwargs: dict = {},
            which_hidden_states: Optional[str]=None,
            **tokenizer_args,):
        """Train a RepReader on the training data.
        Args:
            batch_size: batch size to use when getting hidden states
            direction_method: string specifying the RepReader strategy for finding directions
            direction_finder_kwargs: kwargs to pass to RepReader constructor
        """

        if not isinstance(hidden_layers, list): 
            assert isinstance(hidden_layers, int)
            hidden_layers = [hidden_layers]
        
        self._validate_params(n_difference, direction_method)

        # initialize a DirectionFinder
        direction_finder = DIRECTION_FINDERS[direction_method](**direction_finder_kwargs)

		# if relevant, get the hidden state data for training set
        hidden_states = None
        relative_hidden_states = None
        if direction_finder.needs_hiddens:
            # get raw hidden states for the train inputs
            hidden_states = self._batched_string_to_hiddens(train_inputs, rep_token, hidden_layers, batch_size, which_hidden_states, **tokenizer_args)
            
            # get differences between pairs
            relative_hidden_states = {k: np.copy(v) for k, v in hidden_states.items()}
            for layer in hidden_layers:
                for _ in range(n_difference):
                    relative_hidden_states[layer] = relative_hidden_states[layer][::2] - relative_hidden_states[layer][1::2]

		# get the directions
        direction_finder.directions = direction_finder.get_rep_directions(
            self.model, self.tokenizer, relative_hidden_states, hidden_layers,
            train_choices=train_labels)
        for layer in direction_finder.directions:
            if type(direction_finder.directions[layer]) == np.ndarray:
                direction_finder.directions[layer] = direction_finder.directions[layer].astype(np.float32)

        if train_labels is not None:
            direction_finder.direction_signs = direction_finder.get_signs(
            hidden_states, train_labels, hidden_layers)
        
        return direction_finder
