# PINN-SPM

## Physics-Informed Neural Networks (PINNs) for single particle battery model (SPM) parameter identification and state estimation. 

### Method
The network consists of neural networks (NNs) and a battery model. For the NNs, recurrent neural networks, namely long short-term memory (LSTM) networks, are used. The LSTM inputs historical battery operational data (current and voltage) and outputs the battery's internal states, i.e. lithium-ion concentration at each electrode. The battery model takes the outputs and calculates other battery variables, such as voltage. Over the training, the networks and battery model parameters are optimized such that predicted internal states and voltage outputs align with the physical law of the battery model and the observed voltage. For the methodology in detail, please refer to [this publication](https://ieeexplore.ieee.org/document/10644822)(a simpler NN and battery model are used in the paper).

### Features
- Single Particle Model(SPM) is used for the battery model
- Six model parameters (e.g. diffusion coefficient at the anode) can be set to be identified (over the NNs training)
- After the NNs are trained, they can be used as rapid state estimators
- Two cell models (fresh and aged) included
- Five cycling datasets are prepared using [SPMe code](https://github.com/scott-moura/SPMeT) from Prof. Scott Moura (with electrolyte dynamics disabled)
- Transfer learning concept implemented: before training PINN for the cell with unknown parameters (e.g. the degraded cell), NN is pre-trained with the simulated dataset (that includes internal states) of the cell with known parameters (e.g. the fresh cell). It allows faster and stabler PINN training.

### Usage
To train the model, run:

python main.py config_pinn.yaml

To change the configuration (e.g. parameters to be identified, datasets to be used), edit config_pinn.yaml.

If you want to change network architecture (e.g., the size of the hidden layer of the LSTM), you will first need to edit config_nn.yaml and redo NN pre-training by running:

python main.py config_nn.yaml

