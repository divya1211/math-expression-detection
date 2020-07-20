import torch
from transformers import BertForSequenceClassification, BertTokenizer
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import AdamW

from collections import Counter
from pathlib import Path
from data import MathExpressionsDataset
from model import RobertaClassificationHead


def train(epochs=10, bs=32, lr=1e-6):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = MathExpressionsDataset()
    dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True)
    
    dist = Counter(y for x, y in train_dataset)
    print(f"The dataset size is {len(train_dataset)} and {dist}")

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir="./models")
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', cache_dir="./models")
    
    # tokenizer = RobertaTokenizer.from_pretrained('roberta-base', cache_dir="./models")
    # model = RobertaForSequenceClassification.from_pretrained('roberta-base', cache_dir="./models")
    # model.classifier = RobertaClassificationHead()
    
    model = model.to(device)
    model.train()

    # optimizer
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)

    for epoch in range(epochs):
        for idx, (text_batch, label) in enumerate(dataloader):
            encoding = tokenizer(text_batch, return_tensors='pt', padding=True, truncation=True)
            input_ids = encoding['input_ids']
            attention_mask = encoding['attention_mask']
            labels = label.unsqueeze(0)
            
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            loss.backward()
            optimizer.step()

            if idx % 10 == 0:
                print(f"Epoch is {epoch} and Loss is {round(loss.item(), 2)}.")    

    model.save_pretrained('./models/math-classifier')


if __name__ == "__main__":
    train()

