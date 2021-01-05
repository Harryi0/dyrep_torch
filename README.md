# dyrep_torch

Main structure and data processing refer to: https://github.com/uoguelph-mlrg/LDG

**SocialEvolution Dataset**: 

Time Prediction MAE (5 epochs): 16.48 (hrs); Paper: 23.44 (hrs)

``` shell script
python train_eval_main.py --data_dir ./SocialEvolution --dataset social
```

**Wikipedia Dataset**: 

Time Prediction MAE (4 epochs): 21.09 (hrs)

``` shell script
python train_eval_main.py --data_dir ./Jodie --dataset wikipedia --all_comms True
```

**Reddit Dataset**: 

Time Prediction MAE (2 epochs): 28.22 (hrs)

``` shell script
python train_eval_main.py --data_dir ./Jodie --dataset reddit --all_comms True
```

**Synthetic Dataset**: Generate from Hawkes process (TPP) with  20 nodes and 10,000 events (all communication) 
accoding to the method from: 
[Omi, Takahiro, and Kazuyuki Aihara. "Fully neural network based model for general temporal point processes." Advances in Neural Information Processing Systems. 2019.]

Time Prediction MAE (n epochs): 292.19 (hrs)

``` shell script
python train_eval_main.py --data_dir ./Jodie --dataset reddit --all_comms True
```