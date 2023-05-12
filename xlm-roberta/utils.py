import datasets
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
def preprocess_function(examples):
        # Tokenize the texts
        return tokenizer(
            examples["premise"],
            examples["hypothesis"],
            padding="max_length",
            max_length=64,
            truncation=True,
            return_tensors='pt'
        )

def get_datasets():
    # Get train, test, and validation datasets for each language
    xnli_dataset_en = datasets.load_dataset('xnli', language='en')
    xnli_dataset_de = datasets.load_dataset('xnli', language='de')
    xnli_dataset_fr = datasets.load_dataset('xnli', language='fr')

    train_en = xnli_dataset_en['train']
    train_de = xnli_dataset_de['train']
    train_fr = xnli_dataset_fr['train']

    test_en = xnli_dataset_en['test']
    test_de = xnli_dataset_de['test']
    test_fr = xnli_dataset_fr['test']

    val_en = xnli_dataset_en['validation']
    val_de = xnli_dataset_de['validation']
    val_fr = xnli_dataset_fr['validation']

    # Concatenate datasets for all languages and shuffle datasets
    SEED = 89
    train_dataset = datasets.concatenate_datasets([train_en, train_de, train_fr]).shuffle(seed=SEED)
    test_dataset = datasets.concatenate_datasets([test_en, test_de, test_fr]).shuffle(seed=SEED)
    val_dataset = datasets.concatenate_datasets([val_en, val_de, val_fr]).shuffle(seed=SEED)

    return train_dataset, test_dataset, val_dataset