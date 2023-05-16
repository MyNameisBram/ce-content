import streamlit as st

import streamlit as st
import openai
import pinecone

# Set up OpenAI API credentials

def set_openai_api_key(api_key: str):
    st.session_state["OPENAI_API_KEY"] = api_key

def set_pinecone_api_key(api_key: str):
    st.session_state["PINECONE_API_KEY"] = api_key


# Define function to convert input prompt to OpenAI embedding
def embed_prompt(prompt):
    embeddings = openai.Embedding.create(
        input=[prompt],
        engine="text-embedding-ada-002"
    )
    embedding = embeddings['data'][0]['embedding']
    return embedding



# Define function to generate custom prompt
def generate_prompt(disc_type, objective_verb, product_type):

    text = "What's the best way to communicate to a disc type {} if I'm trying to {} to them {}? ".format(disc_type, objective_verb, product_type)

    prompt = f"""
    You will be provided with text delimited by triple quotes. 
    When answering the question, do not use languge that is too technical or confusing.
    
    - First, answer the question using a minimum number of words while still maintaining clarity and effectiveness. 
    See below for an example, but do not show the word "Answer:" :

    example: 
    <Question> What's the best way to communicate to a disc type Di ?
    <Answer> Focus on big ideas, big concepts, and fast action (over analysis).

    - Second, Provide two bullet points each of do's and don'ts 

    Do
    - <content>
    - <content>

    Don't
    - <content>
    - <content>
    \"\"\"{text}\"\"\"
    """

    return prompt
  
  
def generate_text(disc_type, verb, product, api_key_input):
    # Get input prompt from request
    #disc_type = request.json['disc_type'] # like Di, iD, SC, CS
    #verb = request.json['verb'] # like sell, convince, persuade
    #product = request.json['product'] # like a product, a service, a solution
    openai.api_key = api_key_input

    prompt = generate_prompt(disc_type, verb , product)

    # Embed input prompt using OpenAI API
    embedding = embed_prompt(prompt)

    # Query Pinecone index to find top_k=5
    query_result = pinecone_index.query(embedding, top_k=5, include_metadata=True)

    # add context to prompt
    contexts = [item['metadata']['text'] for item in query_result['matches']]

    # Add context from Pinecone to create augmented prompt
    augmented_prompt = "\n\n---\n\n".join(contexts)+"\n\n-----\n\n"+prompt

    # system message to 'prime' the model
    primer = f"""You are content generator bot. A highly intelligent system that answers
    user request based on the information provided by the user above
    each question. If the information can not be found in the information
    provided by the user you truthfully say "I don't know".
    """

    # Generate text using augmented prompt and GPT-3.5 Turbo model
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": primer},
            {"role": "user", "content": augmented_prompt}
        ]
    )

    return response.choices[0]['message']['content']
  

   
    

# display main
def main():

    api_key_input = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="Paste your OpenAI API key here (sk-...)",
            help="You can get your API key from https://platform.openai.com/account/api-keys.",  # noqa: E501
            value=st.session_state.get("OPENAI_API_KEY", ""),)
    
    #openai.api_key = api_key_input
    

    # pinecone keys
    PINECONE_API_ENV = "asia-southeast1-gcp"

    pinecone_api_key_input = st.text_input(
            "PineconeDB API Key",
            type="password",
            placeholder="Paste your PineconeDB API key here (sk-...)",
            value=st.session_state.get("PINECONE_API_KEY", ""),)
    
    # init pinecone
    # Set up PineconeDB credentials
    pinecone.init(
        api_key=pinecone_api_key_input,  # find at app.pinecone.io
        environment=PINECONE_API_ENV  # next to api key in console
    )
    index_name = "crystal-disc" # put in the name of your pinecone index here
    # connect to index
    pinecone_index = pinecone.Index(index_name)
    

    # Create a dropdown menu for the first input
    option1 = st.selectbox("Select Option 1", 
        ['IS', 'Is', 'I', 'Id', 'DI', 'Di', 'D', 'Dc', 'CD', 'Cd', 'C', 'Cs', 'SC', 'Sc', 'S', 'Si'])
    st.write("You selected:", option1)
    
    # Create a dropdown menu for the second input
    option2 = st.selectbox("Select Option 2", 
        ["sell", "email", "call"])
    st.write("You selected:", option2)
    
    # Create a dropdown menu for the third input
    option3 = st.selectbox("Select Option 3", 
        ["a product", "a Saas product", "a pen"])
    st.write("You selected:", option3)

    # Create a button to run the LLM generator
    if st.button("Generate Content"):
        # run LLM generator 
        result = generate_text(option1, option2, option3, api_key_input)
        st.write("LLM generated content:")
        st.write(result)



if __name__ == "__main__":
    main()
