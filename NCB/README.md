
# Setup
1. Install svox2-fast
2. Download LLFF & nerf-synthetic dataset
3. Set the dataset setting in the **autotask.py**

# Training


launch nerf-synthetic experiments
```python
python autotask -g "0 1 2 3 4 5 6 7" -llff
```

launch LLFF experiments
```python
python autotask -g "0 1 2 3 4 5 6 7" -llff
```

# Testing



launch nerf-synthetic experiments
```python
python autotask -g "0 1 2 3 4 5 6 7" --syn --eval
```

launch LLFF experiments
```python
python autotask -g "0 1 2 3 4 5 6 7" --llff --eval
```
