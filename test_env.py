import torch
import transformers
import datasets
import sklearn
import pandas
import numpy

print("Torch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("Datasets:", datasets.__version__)
print("MPS Available:", torch.backends.mps.is_available())