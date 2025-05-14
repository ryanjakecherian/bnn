bnn
===

Training Binary Neural Network with the Bernouilli Optimiser


Environment set-up
--------

```
conda env create --file envs/environment.yaml
conda activate [NAME_OF_ENV (should be bnn)]
pip install -e /path/to/this/folder/
```

If you're developing:
```
pre-commit install
```

Running the Experiments
--------
To run the experiment run the following file.
```
bnn/src/scripts/train_classification.py
```

To adjust the network hyperparameters, edit the following files.
```
bnn/config/main.yaml
bnn/config/network/TBNN_mnist.yaml
bnn/config/W_optimizer/ExpectationSGD.yaml
```
