# PINN-SPM

## Physics-Informed Neural Networks (PINNs) for single particle battery model (SPM) parameter identification and state estimation. 

### Method
The network consists of neural networks (NNs) and a battery model. For the NNs, recurrent neural networks, namely long short-term memory (LSTM) networks, are used. The LSTM inputs historical battery operational data (current and voltage) and outputs the battery's internal states, i.e. lithium-ion concentration at each electrode. The battery model takes the outputs and calculates other battery variables, such as voltage. 

### Features
- Single Particle Model(SPM) is used for the battery model
- Six model parameters (e.g. diffusion coefficient at anode) can be set to be identified
- Two cell models (fresh and aged) included
- Five cycling datasets are prepared using [SPMe code](https://github.com/scott-moura/SPMeT) from Prof. Scott Moura (with electrolyte dynamics disabled)

### Usage
To train the model, run:

python main.py config_pinn.yaml
