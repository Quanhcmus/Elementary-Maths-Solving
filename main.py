from preprocessing_data import preprocessing, processing_answer
from fine_tunning import fine_tunning
from pipeline import pipeline

def fine_tunning_model():
    preprocessing("data/math_train.json", "data/train.csv")
    preprocessing("data/math_test.json", "data/test.csv", is_train=False)
    fine_tunning(base_model_path="model/ura_llama2_7b", new_model_path = "model/ura_llama2_7b_new")
    
if __name__ == '__main__':
    print("Đang fine tunning model đợi khoảng 30p...")
    fine_tunning_model()
    print("Đã fine-tunning xong. Đợi merge base model và new model...")
    pipe = pipeline(base_model_path="model/ura_llama2_7b", new_model_path = "model/ura_llama2_7b_new")
    query_template = "[INST Hãy trả lời câu hỏi sau và chọn đáp án đúng từ các đáp án đã cho.]{query}, các đáp án{da}[/INST]"
    query = "Hỗn số 68 $\\frac{7}{100}$  dưới dạng số thập phân là:"
    da = "\n68,007\n68,07\n68,7\n68,70"
    # example
    while True:
        print('Đây là câu hỏi mẫu:', query, da)
        query_s = query_template.format(query=query,da = da)
        answer = pipe(query_s)[0]["generated_text"]
        print('Đáp án là:',processing_answer(answer),sep='\n')
        a = input("Bạn còn câu hỏi nào không(nhập 0 nếu muốn dừng):")
        if a == 0:
            break
        else:
            query = input('Mời bạn nhập câu hỏi:')
            da = input('Mời bạn nhập các đáp án:')
    