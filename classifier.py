import sys
import torch
from transformers import BertForSequenceClassification, BertTokenizer
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from model import RobertaClassificationHead


class Classifier():
    def __init__(self, model_path='./models/math-classifier'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_classifier(model_path)

    
    def load_classifier(self, model_path):
        self.model = BertForSequenceClassification.from_pretrained(model_path, cache_dir="./models")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", cache_dir="./models")
        
        # self.model = RobertaForSequenceClassification.from_pretrained('roberta-base', cache_dir="./models")
        # self.model.classifier = RobertaClassificationHead()
        # self.model = self.model.from_pretrained(model_path, cache_dir="./models")
        # self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base', cache_dir="./models")
        
        self.model = self.model.to(self.device)


    def ismath(self, text):
        encoding = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
        outputs = self.model(input_ids, attention_mask=attention_mask)
        pred = outputs[0].argmax(1)
        return bool(pred.item())  

if __name__ == "__main__":
    clf = Classifier()
    ip = sys.args[1]
    print(clf.ismath(ip))

