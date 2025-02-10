# PINN-SPM

## Physics-Informed Neural Network(PINN) for battery model parameter identification and state estimation. 

- Single Particle Model(SPM) is used for the battery model
- Six model parameters (e.g. diffusion coefficient at anode) can be set to be identified
- Two cell models (fresh and aged) included
- Five cycling datasets are prepared using [SPMeT code](https://github.com/scott-moura/SPMeT) from Prof. Scott Moura (with electrolyte and thermal dynamics disabled)


To train the model, run:

python main.py config_pinn.yaml
