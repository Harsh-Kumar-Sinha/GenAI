import streamlit as st
from ctransformers import AutoModelForCausalLM, AutoConfig, Config

st.title("DOCKERIZED APIs")

option_list = ["Mistral-7B", "Phi-2"]
selected_option = st.selectbox("Select an option:", option_list)
st.write("Selected option:", selected_option)

# Input box
user_input = st.text_input("Enter something:")

#Check suitable Endpoint
if selected_option == "Phi-2":
    local_llm = "phi-2.Q5_K_S.gguf"
    conf = AutoConfig(Config(temperature=0.2,context_length= 2040))
    llm = AutoModelForCausalLM.from_pretrained(local_llm,model_type="mistral",config = conf)
elif selected_option == "Mistral-7B":
    local_llm = "mistral-7b-instruct-v0.1.Q5_K_S.gguf"
    conf = AutoConfig(Config(temperature=0.2,context_length= 2040))
    llm = AutoModelForCausalLM.from_pretrained(local_llm,model_type="mistral",config = conf)


# Output box
response = llm(user_input,temperature=0.2)

st.text_area("",value=response, height=100)









