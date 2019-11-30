import torch
from transformers import BertTokenizer, BertModel,BertForQuestionAnswering

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
question = ""Your Question
text = "Your Text"
input_text = "[CLS] " + question + " [SEP] " + text + " [SEP]"
input_ids = tokenizer.encode(input_text)
token_type_ids = [0 if i <= input_ids.index(102) else 1 for i in range(len(input_ids))]
start_scores, end_scores = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([token_type_ids]))
all_tokens = tokenizer.convert_ids_to_tokens(input_ids)
print(' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1]))
