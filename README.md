# ETDM

[Python 3.6]
[PyTorch 1.1]

# Look Back and Forth: Video Super-Resolution with Explicit Temporal Difference Modeling (ETDM)

The *official* implementation for the [ETDM] which is accepted by [CVPR 2022].


### Train
We utilize 4 V100 GPUs for training.
```
cd ETDM-CVPR2022
python main.py (add the vimeo dataset path in main.py)

```

### Test
We utilize 1 V100 GPU for testing.
Test the trained model with best performance by
```
cd ETDM-CVPR2022 
python test.py (add the test dataset path in test.py)
```
