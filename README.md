# Elementary-Maths-Solving
Use the large language model ura-llama2 7b to fine-tunning the model from available data on level 1 problems.  
The data is obtained from the Zalo AI Challenge's Elementary Maths Solving competition. This is link to download data [link](https://challenge.zalo.ai/portal/elementary-maths-solving).
## Quickstart
Follow these steps to run the model.
1. Clone this repository  
```
git clone https://github.com/Quanhcmus/Elementary-Maths-Solving
cd Elementary-Maths-Solving
```
2. Download model ura-llama2-7b  
Download the model from the following [link](https://huggingface.co/ura-hcmut/ura-llama-7b) . You need to write a request letter and send it to the owner of the model to obtain permission for downloading.
3. Download data
Download the data from the provided link and save it into the 'data' folder.  
The structure of the entire source code is saved as follows:
```
Elementary-Maths-Solving/
|
├── data/
|   ├── math_train.json
|   └── math_test.json
|
├── model/
|   ├── ura_llama2_7b/
|   └── ura_llama2_7b_new/
|
├── fine_tunning.py
|
├── main.py
|
├── pipeline.py
|
├── preprocessing_data.py
│
└── README.md
```
3. Run this model  
```
!pip install -q accelerate==0.21.0 peft==0.4.0 bitsandbytes==0.40.2 transformers==4.31.0 trl==0.4.7
!python main.py
```
## Code overview
+ The data preprocessing for preparing the fine-tuning process is all included in the **preprocessing_data.py** file.   
    The data, which is in a JSON file, reads information from the 'data' folder, processes it, and then writes the modified data into 'data/train.csv' and 'data/test.csv'.
+ The model fine-tuning process from the preprocessed data is executed within the **fine_tuning.py** file.  
With the defined parameters for fine-tuning using the RoLa technique, the model will be fine-tuned and saved to the specified path.
+ The **pipeline.py** is used to merge the just-saved model with the pre-downloaded 'ura-llama-7b' model to answer the provided questions.
+ Finally, the **main.py** file is used to execute the entire process from data preprocessing to answering questions.
## reference
+ Model ura-llama2-7b: https://huggingface.co/ura-hcmut/ura-llama-7b
+ LoRa fine tunning: https://deci.ai/blog/fine-tune-llama-2-with-lora-for-question-answering/