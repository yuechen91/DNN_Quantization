import torch
import struct
from transformers import BertConfig, BertTokenizer, BertForQuestionAnswering
import re

def Diff(li1, li2):
    li_dif = [i for i in li1 + li2 if i not in li1 or i not in li2]
    return li_dif

def process_name(name):
    temp = re.findall(r'\d+', name)
    if len(list(map(int, temp))) == 0 :
        return name
    else:
        return list(map(int, temp))[0]

def print_tensor_in_hex(param, shape, f):
    if len(shape) > 1 :
        for i in range(0,shape[0]):
            for j in range(0,shape[1]):
                print(struct.pack("f", float(param[i][j])).hex(), file=f)
    else :
        for i in range(0,shape[0]):
            print(struct.pack("f", float(param[i])).hex(), file=f)

def print_tensor_in_float(param, shape, f):
    if len(shape) > 1 :
        for i in range(0,shape[0]):
            for j in range(0,shape[1]):
                print(float(param[i][j]), file=f)
    else :
        for i in range(0,shape[0]):
            print(float(param[i]), file=f)

def get_bert_qa_model(model_name="deepset/bert-base-cased-squad2", cache_dir="./saved_models"):
    # https://huggingface.co/transformers/model_doc/bert.html#transformers.BertForQuestionAnswering
    config = BertConfig.from_pretrained(model_name, cache_dir=cache_dir)
    tokenizer = BertTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = BertForQuestionAnswering.from_pretrained(model_name, config=config, cache_dir=cache_dir)

    return model, tokenizer


model, tokenizer = get_bert_qa_model(model_name="deepset/bert-base-cased-squad2")

quantized_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

layer_list = []
qlayer_list = []

for name, param in model.named_parameters():
    layer_list.append(name)


for name, param in quantized_model.named_parameters():
    qlayer_list.append(name)

layer_name = Diff(layer_list,qlayer_list)

l = int(0)
old_l = int(0)
bert = open('bert.layer0.txt','w')
bert_float = open('bert.layer0.float.txt','w')
print("=" * 75)
for name, param in model.named_parameters():
    if name in layer_name:
        l = process_name(name)
        if l != old_l:
            bert.close()
            print("=" * 75)
            bert_float.close()
            file_name = 'bert.layer{number}.txt'.format(number=l)
            file_name_float = 'bert.layer{number}.float.txt'.format(number=l)
            if isinstance(l,str):
                file_name = 'bert.{number}.txt'.format(number=l)
                file_name_float = 'bert.{number}.float.txt'.format(number=l)
            bert = open(file_name,'w')
            bert_float = open(file_name_float,'w')
        old_l = l
        print(name,file=bert)
        print(param.shape,file=bert)
        print_tensor_in_hex(param,param.shape,bert)
        print(name,file=bert_float)
        print(param.shape,file=bert_float)
        print_tensor_in_float(param,param.shape,bert_float)
        print(name)
        print(param.shape)

print("=" * 75)
bert.close()
bert_float.close()
