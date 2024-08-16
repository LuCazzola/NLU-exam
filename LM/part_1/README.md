# Task description (lab 4, part 1)
In this, you have to modify the baseline LM_RNN by adding a set of techniques that might improve the performance. In this, you have to add one modification at a time incrementally.
For each of your experiments, you have to print the performance expressed with Perplexity (PPL).

* Replace RNN with a Long-Short Term Memory (LSTM) network
* Add two dropout layers
  * one after the embedding layer
  * one before the last linear layer
* Replace SGD with AdamW

## Note for professors
Yhe project uses **wandb** as logger to store the metrics.
If you've already installed the requirements of the [**course labs**](https://github.com/BrownFortress/NLU-2024-Labs), simply install the dependancy for wandb
```
pip install wandb
```

# Usage
## Train model

```
```

Also a bash script is available to make arguments setting easier :
```
chmod +x run.sh
./run.sh
```

Otherwise, try :
```
python3 main.py ...
```

## Test best models
```
python3 main.py ...
```
