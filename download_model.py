from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM

# Specify the model name you want to use
# model_name = "openai-community/gpt2"
model_name = "facebook/opt-125m"

# Download a tokenizer object by loading the pretrained "Intel/dynamic_tinybert" tokenizer.
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Download a question-answering model object by loading the pretrained "Intel/dynamic_tinybert" model.
model = AutoModelForCausalLM.from_pretrained(model_name)