import torch
import numpy as np

class FenceDataSet(torch.utils.data.Dataset):
    def __init__(self, tokens: dict, fence_dict: dict[str, int], D: int, feature_classifications: list[dict[str, int]], position_mask_start_token_id: int | list[int] | None = None):
        """
        Creates a new FENCE Dataset object
        
        Description:
            Let I = # of examples to pass into the dataset; N = token length per example
        
        Params: 
            @tokens: A dict containing two keys: `input_ids` containing an I x N tensor; and `attention_mask` containing an I x N tensor
            @fence_dict: A dict of features and their corresponding fence dimensions, e.g. {'dogs': (3065, 3068), 'cats': (3061, 3064)}. 
             - These dimensions are 1-indexed and inclusive of both the start and ending numbers passed into the tuples. (3061, 3064) means dimensions 3061, 3062, 3063, and 3064.
            @D: The hidden state dimension
            @feature_classifications: An I-length list of dicts where each dict is a target class [{'cat': 1, 'dog': 0'}, {'cat': 0, 'dog': 1}]
            @position_mask_start_token_id: A token ID. When creating the FENCE mask for each sentence, each token is assigned a 0 or 1. 
             - Tokens are assigned a 0 if they are 0 within the attention mask (i.e., a padding token), or are values that occur BEFORE or EQUAL to the first instance
               of the target token ID (or sequence of token IDs) in the sentence. 
             - Tokens are assigned a 1 if they are 1 within the attention mask (not a padding token) and occur STRICTLY AFTER the first instance of the target token ID 
               in the sentence, OR the sentence has no occurrence of the target token ID at all.
             If this parameter is set to None, the attention mask is re-used as the FENCE target mask.
        """
        if (tokens['input_ids'].shape[0] != len(feature_classifications)):
            raise ValueError('Length of feature_classifications must be the same length of tokens')

        if len({len(d) for d in feature_classifications}) != 1:
            raise ValueError('Length of feature_classifications is not equal')

        if len(feature_classifications[0]) != len(fence_dict):
            raise ValueError('Length of feature_classifications dicts are not equal to length of feature_dict')
            
        if sorted(fence_dict.keys()) != sorted(feature_classifications[0].keys()):
            raise ValueError('Incorrect keys in feature_classifications')
        
        self.F = len(fence_dict)
        self.I = tokens['input_ids'].shape[0]
        self.N = tokens['input_ids'].shape[1]
        
        # Get a D-dimension Dfmask which shows 1 for regions where FENCE is active
        self.Df = np.sum([v[1] - v[0] + 1 for _, v in fence_dict.items()])
        
        self.tokens = tokens
        self.fence_dict = fence_dict
        self.feature_classifications = feature_classifications
        self.feature_names = list(fence_dict.keys())

        self.feature_classifications_tensor = torch.tensor([
            [feat_classes[k] for k, v in fence_dict.items()]
            for feat_classes in feature_classifications
        ]).to(tokens['input_ids'].device)
        
        feature_targets_list = []
        for feat_classes in feature_classifications:
            zeros = torch.zeros(D)
            for k, v in fence_dict.items():
                if feat_classes[k] == 1:
                    zeros[fence_dict[k][0] - 1:fence_dict[k][1]] = 1
            feature_targets_list.append(zeros)

        self.feature_targets = torch.stack(feature_targets_list, dim = 0).to(tokens['input_ids'].device)

        # Set position mask - I x N mask of which tokens should be counted in position loss
        # 9/28/24 - Previously the mask only started at token 2 when completely missing the sequence! Now starts at 0th token when completely missing
        #    switched start_fence_mask[~start_fence_mask_bool.any(dim = 1)]=0 to start_fence_mask[~start_fence_mask_bool.any(dim = 1)]=-1
        if position_mask_start_token_id == None:
            self.position_mask = tokens['attention_mask']
        elif isinstance(position_mask_start_token_id, int):
            # Get first instance of the position_mask_start_token_id in each row
            start_fence_mask_bool = tokens['input_ids'] == position_mask_start_token_id
            start_fence_mask = start_fence_mask_bool.int().argmax(dim = 1)
            start_fence_mask[~start_fence_mask_bool.any(dim = 1)] = -1 # If no occurrence of the value, argmax returns 0, so set those to 0
            # Create a mask which equals 1 for each row for all column values > the first instance of the start FENCE token AND the attention mask == 1
            # Note: the second condition in the logical_and has no effect if padding results in right-aligned text
            self.position_mask = torch.where(
                torch.logical_and(
                    torch.arange(self.N, device = tokens['input_ids'].device).expand(self.I, self.N) >= start_fence_mask.unsqueeze(1) + 1,
                    tokens['attention_mask'] == 1
                ),
                torch.ones_like(tokens['input_ids']),
                torch.zeros_like(tokens['input_ids'])
            )
        elif isinstance(position_mask_start_token_id, list) and all(isinstance(x, int) for x in position_mask_start_token_id):    
            seq_len = len(position_mask_start_token_id)
            # Get a tensor version of position_mask_start_token_id for comparison
            start_fence_tensor = torch.tensor(position_mask_start_token_id, device = tokens['input_ids'].device)
            # Create a sliding window over tokens['input_ids'] to find consecutive matches of position_mask_start_token_id
            # We need to use unfold to create a rolling window over each row.
            unfolded = tokens['input_ids'].unfold(dimension = 1, size = seq_len, step = 1)  # [batch_size, new_seq_len, seq_len]
            # Compare each window with position_mask_start_token_id
            match_mask = (unfolded == start_fence_tensor.unsqueeze(0).unsqueeze(0)).all(dim=2)  # [batch_size, new_seq_len]
            # Get the index of the first occurrence of the exact sequence in each row
            first_fence_mask = match_mask.int().argmax(dim=1)  # Find the first occurrence of the sequence            
            # Handle cases where the sequence doesn't appear by setting them to 0
            first_fence_mask[~match_mask.any(dim=1)] = -1  # If no occurrence of the sequence, set to 0
            # Adjust first_fence_mask to point to the last element of the first found sequence
            first_fence_mask += seq_len - 1  # Shift the mask to point to the end of the matched sequence
            # Create a mask which equals 1 for each row for all column values > the first instance of the start FENCE token AND the attention mask == 1
            self.position_mask = torch.where(
                torch.logical_and(
                    torch.arange(self.N, device=tokens['input_ids'].device).expand(self.I, self.N) >= first_fence_mask.unsqueeze(1) + 1,
                    tokens['attention_mask'] == 1
                ),
                torch.ones_like(tokens['input_ids']),
                torch.zeros_like(tokens['input_ids'])
            )
        else:
            raise ValueError("position_mask_start_token_id should be None, an integer, or a list of integers")

        
    def __len__(self):
        return len(self.tokens['input_ids'])

    def __getitem__(self, idx):
        """
        feature_targets is a B x D_f size object
        """
        return {
            'input_ids': self.tokens['input_ids'][idx],
            'attention_mask': self.tokens['attention_mask'][idx],
            'feature_classifications': self.feature_classifications[idx],
            'feature_targets': self.feature_targets[idx],
            'position_mask': self.position_mask[idx]
        }
    
