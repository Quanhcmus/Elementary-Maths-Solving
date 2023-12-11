import  pandas as pd
import numpy as np
import re

########################################################################################
def read_data(path:str):
    df = pd.read_json(path)
    df = df['data'].apply(pd.Series)
    if "Unnamed: 0" in df.columns:
        df.drop("Unnamed: 0",axis=1,inplace=True)
    return df

def remove_choice(text:str):
    return text[3:]

def markdown_to_plain_text(text):
    if ~pd.isna(text):
        text = str(text)
        # Thay thế các ký tự Markdown phổ biến bằng dạng thường
        text = re.sub(r'(\d+,\d+) / (\d+,\d+)', r'$\\\\frac{\1}{\2}$', text)
        text = re.sub(r'(\d)\^{(\d+)}', r'$\1^\2$', text)
        text = re.sub(r'\^{(\d+)}', r'$^\1$', text)
        text = re.sub(r'(\d) (\d)', r'\1\2', text)
        text = re.sub(r'(\d+) / (\d+)', r'$\\frac{\g<1>}{\g<2>}$', text)
        text = re.sub(r'(\d+)/(\d+)', r'$\\frac{\1}{\2}$', text)
        text = text.replace("­-", "-")
        text = text.replace('{*}', '${\\times}$')
        return text
    return text

def markdown_answer(arr):
    l = []
    for i in arr:
        l.append(markdown_to_plain_text(i))
    return l

def convert_text_train(record:pd.Series):
    
    A = remove_choice(record['choices'][0])
    B = remove_choice(record['choices'][1])
    C = remove_choice(record['choices'][2])
    explanation = record['explanation']
    query = record['question']
    answer = remove_choice(record['answer'])
    if len(record['choices']) ==4:
        D = remove_choice(record['choices'][3])
        text_train_template = "<s>[INST]{query}, các đáp án\n{A}\n{B}\n{C}\n{D}[/INST]</s>\nLời giải:{explanation}\nChọn đáp án:{answer}"
        return text_train_template.format(query = query,explanation = explanation,  answer = answer,A = A, B = B, C = C, D = D) 
    else:
        text_train_template = "<s>[INST]{query}\n{A}\n{B}\n{C}[/INST]</s>\nLời giải:{explanation}\nchọn đáp án:{answer}"
        return text_train_template.format(query = query,explanation = explanation,  answer = answer,A = A, B = B, C = C)
#####################################################################################################################################
def convert_text_test(record:pd.Series):

    A = remove_choice(record['choices'][0])
    B = remove_choice(record['choices'][1])
    C = remove_choice(record['choices'][2])
    query = record['question']
    if len(record['choices']) ==4:
        D = remove_choice(record['choices'][3])
        query_template = "[INST Hãy trả lời câu hỏi sau và chọn đáp án đúng từ các đáp án đã cho.]Câu hỏi:{query} các đáp án\n{A}\n{B}\n{C}\n{D}[/INST]\nĐáp án đúng là:"
        return query_template.format(query = query, A = A, B = B, C = C, D = D)
    else:
        query_template = "[INST Hãy trả lời câu hỏi sau và chọn đáp án đúng từ các đáp án đã cho.]Câu hỏi:{query} các đáp án\n{A}\n{B}\n{C}[/INST]\nĐáp án đúng là:"
        return query_template.format(query = query,A = A, B = B, C = C)
def answer_text(arr):
    l = []
    for i in arr:
        l.append(markdown_to_plain_text(i))
    return l
#####################################################################################################################################

def preprocessing(path_data, csv_path_out, is_train = True):
    df = read_data(path_data)
    if (is_train):
        df['explanation'] = df['explanation'].apply(markdown_to_plain_text)
        df['question'] = df['question'].apply(markdown_to_plain_text)
        df['answer'] = df['answer'].apply(markdown_to_plain_text)
        text = df.apply(convert_text_train,axis=1)
        df_text = pd.DataFrame(
            {
            'text' : text.to_list()
            }
        )
        df_text.to_csv(csv_path_out)
    else:
        df['question'] = df['question'].apply(markdown_to_plain_text)
        df['choices'] = df['choices'].apply(answer_text)
        df = df.apply(convert_text_test,axis=1)
        df.to_csv(csv_path_out)
        
def processing_answer(text:str):
    answer = text.split('\n')
    return answer[0]
        