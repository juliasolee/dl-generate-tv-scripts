## TV Script Generation

This repository contains my solution for Udacity's third Deep Learning Nanodegree project. 
This project generates new [Seinfeld](https://en.wikipedia.org/wiki/Seinfeld) TV scripts using RNNs. The network is trained using part of the [Seinfeld dataset](https://www.kaggle.com/thec03u5/seinfeld-chronicles#scripts.csv) of scripts from 9 seasons. The Neural Network built in this project will generate a new, "fake" TV script, based on patterns it recognizes in this training data.

### Pre-processing

One of the main objectives of the project is to pre-process the data, implementing a LookUp Table and a function to Tokenize Punctuation in the text.

1. **LookUp Table:** to create a word embedding, we first need to transform the words to ids. The `create_lookup_tables` function creates two dictionaries `vocab_to_int` and `int_to_vocab` to go from words to id and backwards.

2. **Tokenize Punctuation:** the network must tokenize symbols like "." into "||period||" to avoid creating multiple ids for same words (e.g. "bye" and "bye!"). The `token_lookup` function creates a dictionary that is used to tokenize the symbols and add the delimiter (space) around it. This separates each symbols as its own word, making it easier for the neural network to predict the next word.

### Training the model

- **Batching**
Implementation of the `batch_data` function to batch `words` data into chunks of size `batch_size` using the [TensorDataset](https://pytorch.org/docs/master/data.html#torch.utils.data.TensorDataset) and [DataLoader](https://pytorch.org/docs/master/data.html#torch.utils.data.DataLoader) classes.

- **Building the Neural Network**
Implementation of a RNN using PyTorch's [Module class](https://pytorch.org/docs/master/nn.html#torch.nn.Module) using LSTM and use it to apply forward and back propagation. The `forward_back_prop` function is called iteratively in the training loop and returns the average loss over a batch and the hidden state returned by a call to the RNN class.

- **Hyperparameters**

We aimed for a loss less than 3.5. The value of the hyperparameters was chosen and modified in order to achieve this goal.

- `Sequence length`: it stared being 20 initially and I decreased it to 10 due to the increasing of loss. The longer the sequence length, the more difficulties to learn. Training on smaller sequences provides a significant initial speed up in loss reduction. With the sequence length we need to divide the batches. Finally, we were using batch_size=128, so we chose sequence_length=8.

- `Batch size`: Started at 50. After training, I increased this number to 128 as it took too long and was not efficient.

- `Number of epochs`: 20 epochs were enough to meet the requirements. Moreover, 12 would have been enough.

- `Learning rate`: I started testing at 0.01 and modify it looking at the loss when training. The loss remained stuck between the same values, I decreased it to 0.001.

- `Hidden dimension`: usually larger is better performance wise. Common values are 128, 256, 512, etc. I have used 256, otherwise it would have taken much longer.

- `Embedding dimension`: The latent dimension of the embedding layer should be "as large as possible". Typical values for a dictionary around our size can be 128 or 256. Therefore, 256 was chosen.

- `Number of layers`à we usually use between 1-3. As explained in the lesson, for RNNs using between 1-3 rnn layers there is difference but going deeper rarely helps much more. I decided to use 2 hidden layers.

For more implementation details, have a look [here](https://github.com/juliasolee/dl-generate-tv-scripts/blob/main/dlnd_tv_script_generation.ipynb).

