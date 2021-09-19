# !pip install datasets
from datasets import load_dataset
dataset = load_dataset('winograd_wsc', 'wsc285')
print(dataset)
# for i in dataset['test']:
#     print(i)