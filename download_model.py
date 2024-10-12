from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# Specify the model name you want to use
# model_name = "openai-community/gpt2"
model_name = "deepset/tinyroberta-squad2"

# Download a tokenizer object by loading the pretrained "Intel/dynamic_tinybert" tokenizer.
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Download a question-answering model object by loading the pretrained "Intel/dynamic_tinybert" model.
model = AutoModelForQuestionAnswering.from_pretrained(model_name)