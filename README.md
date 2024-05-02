# speech-recognition-snn

## Dataset

[https://github.com/Jakobovski/free-spoken-digit-dataset](https://github.com/Jakobovski/free-spoken-digit-dataset)

Put audio files in `.data/recordings`

## Environment

Create environment using `python3 -m venv .venv`

Activate environment with `source .venv/bin/activate`

Install dependencies with `pip install -r requirements.txt`

## Run
```bash
python3 snn.py
```

## Sample output
```
Loading data...
3000 samples
device cuda
epoch 0
        training loss 2.28
accuracy 0.19
epoch 1
        training loss 2.17
accuracy 0.31
epoch 2
        training loss 1.78
accuracy 0.43
epoch 3
        training loss 1.20
accuracy 0.59
...
epoch 17
        training loss 0.07
accuracy 0.84
epoch 18
        training loss 0.07
accuracy 0.84
epoch 19
        training loss 0.07
accuracy 0.85
```

## Formatting used
```bash
black -l 120 *.py
```