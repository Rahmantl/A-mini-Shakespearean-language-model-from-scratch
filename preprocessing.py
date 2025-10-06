import torch
import tiktoken
import os
#from data import tiny_shakespeare


def get_batches(train_data, val_data, split, batch_size, context_length):
    """
    Gets data samples from the required dataset
    :param train_data:
    :param val_data:
    :param split:
    :param batch_size:
    :param context_length:
    :return:
    """
    data = train_data if split == 'train' else val_data
    ix = torch.randint(0, data.size(0) - context_length - 1, (batch_size,))
    x = torch.stack([data[i:i + context_length] for i in ix]).long()
    y = torch.stack([data[i + 1:i + context_length + 1] for i in ix]).long()
    return x, y


def train_val_split(dataset, ratio):
    """
    Splits the dataset into training and validation sets. The ratio
    :param dataset:
    :param ratio:
    :return:
    """
    n = int(ratio * len(dataset))
    train_data = dataset[:n].clone().detach()
    val_data = dataset[n:].clone().detach()
    return train_data, val_data


def tokenize_train_val_split(method: str):
    """
    The method
    :param method:
    :return:
    """
    if method == "tiktoken":

        #TODO Load the data from data/tiny_shakespeare and encode the text using the tiktoken tokenizer
        # (use r50k\_base) feel free to play around with other pre-trained tokenizers.
        ######################
        # Path to dataset
        base_path = "/content/drive/My Drive/Colab Notebooks/Exercise_1"
        file_path = os.path.join(base_path, "data", "tiny_shakespeare.csv")
        #file_path = r"data\tiny_shakespeare.csv" 
        # Then read the file,  I set the encoding to utf-8 
        with open(file_path, "r", encoding="utf-8") as f:
            dataset = f.read()
            
        # print("length of dataset in characters: ", len(dataset))  
        # print("dataset preview: ", dataset[:1000])
        
          
        ###we can have other tokenizers as well, e.g. cl100k_base, p50k_base, gpt2, etc.   
        tokenizer = tiktoken.get_encoding("r50k_base")
        tokens = tokenizer.encode(dataset) #string to integer
        
        #####

        dataset = torch.tensor(tokens, dtype=torch.long)
        train_data, val_data = train_val_split(dataset, ratio=0.9)
        vocab_size = tokenizer.n_vocab
        decoder = tokenizer

    elif method == "char":
        # TODO Load the data from data/tiny_shakespeare and encode the text using a self-implemented character tokenizer.
        #  The decode function should be a lambda function which selects the corresponding character given a token.
        #  You can also use another implementation, but you may have to change line 87 in the main.py file.
        ######################
        # Path to dataset
        base_path = "/content/drive/My Drive/Colab Notebooks/Exercise_1"
        file_path = os.path.join(base_path, "data", "tiny_shakespeare.csv")
        #file_path = r"data\tiny_shakespeare.csv" 
        # Then read the file, I set the encoding to utf-8 
        with open(file_path, "r", encoding="utf-8") as f:
            dataset0 = f.read()
        print("length of dataset in characters: ", len(dataset0))
        
        #characters = sorted(list(dataset0)) # First we get all the chars in the dataset (I sorted them)
        characters = sorted(list(set(dataset0)))  # Another way to get all the unique chars without repetition  
        #print("characters: ", ''.join(characters))
        ## Now we have the vocabulary size
        vocab_size = len(characters)
        #print("vocab_size: ", vocab_size)
        
        ## Then we can map each char to an integer and vise versa
        str_to_int = {ch: i for i, ch in enumerate(characters)}
        int_to_str = {i: ch for i, ch in enumerate(characters)} 
        
        ## Encoded the whole dataset (string to list[integer])
        encoded_dataset = [str_to_int[ch] for ch in dataset0] 

        ## Decoder function (list[integer] to string)
        decoded_dataset = lambda l: "".join(int_to_str[int(i)] for i in l)
        # the output format will be, for example, {'\n': 0, ' ': 1, '!': 2, '$': 3, '&': 4, ...}
        
        ## And the tokenizer function will be as follows with lambda function
        #tokenizer = lambda i: int_to_str[i]  
        tokenizer = lambda tokens: "".join(int_to_str[int(i)] for i in tokens)
        #####

        dataset = torch.tensor(encoded_dataset, dtype=torch.long)
        # print("dataset shape: ", dataset.shape)
        # print("data_type: ", dataset.dtype)
        train_data, val_data = train_val_split(dataset, ratio=0.9)
    return train_data, val_data, vocab_size, tokenizer
