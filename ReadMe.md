let's clean up LazyControl (ACS)

## Todos
- [ ] Migrate from lazy-vicsek
  - [ ] Adapt env config params
  - [ ] Check control params
  - [ ] Set default configs
- [ ] Integrate lazy-message-listener
  - [ ] Define obs/act spaces
  - [ ] Check obs/act spaces in `__init__`
  - [ ] Update ACS control
  - [ ] Check `env_transition` with laziness for control
  - [ ] Observation (`get_obs`)
  - [ ] Update terminal conditions
  - [ ] Update reward function
  - [ ] Think about the env-compatibility with the previous...
- [ ] Model
  - [ ] Check input output
  - [ ] Check action dist
  - [ ] Check masking method
  - [ ] Check context embedding
  - [ ] Update decoder and generator
  - [ ] Think about the model-compatibility with the previous...
- [ ] ++
- [ ] Scripts for plots
  - [ ] Define data structure for experiments
  - [ ] Data collector
  - [ ] Plot info over time
    - [ ] Entropies (spatial and velocity)
  - [ ] Plot an instance
  - [ ] Create videos
    - [ ] All for 0 to *t*<sub>f</sub>
    - [ ] Each for 0 to *t*<sub>f</sub>

## Assumptions
- Local communication (disc model)
- Parameter sharing (policy)
- Central critic in CTDE
- 

## Dependencies
- `ray==2.1.0`
- `gym==0.23.1`
- `pydantic==1.10.13`


## Environment Parameters

## RL Hyperparameters

## NN Parameters