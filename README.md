# ETDM

# Look Back and Forth: Video Super-Resolution with Explicit Temporal Difference Modeling (ETDM)

The *official* implementation for the [ETDM] which is accepted by [CVPR 2022].

! [framework] (figs/framework.png)

### Train
We utilize 4 V100 GPUs for training.
```
python main.py

```

### Test
We utilize 1 V100 GPU for testing.
Test the trained model with best performance by
```
python test.py
```
