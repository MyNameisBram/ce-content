import streamlit as st
import openai
import pinecone




# pinecone keys

pinecone.init(
    api_key=PINECONE_API_KEY,  # find at app.pinecone.io
    environment=PINECONE_API_ENV  # next to api key in console
)
index_name = "crystal-disc" # put in the name of your pinecone index here
# connect to index
pinecone_index = pinecone.Index(index_name)


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
  
  
def generate_text(disc_type, verb, product):
    # Get input prompt from request
    #disc_type = request.json['disc_type'] # like Di, iD, SC, CS
    #verb = request.json['verb'] # like sell, convince, persuade
    #product = request.json['product'] # like a product, a service, a solution

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
  
  
  
  
 
# predict function
def generate_content():
    st.title("How to Communicate - LLM Generated Content")

    st.write("""### Select 1) disc type 2) verb 3) type of product.""")
    disc_type = st.selectbox(
      'Select Disc Type',
      ('IS', 'Is', 'I', 'Id', 'DI', 'Di', 'D', 'Dc', 'CD', 'Cd', 'C', 'Cs', 'SC', 'Sc', 'S', 'Si') 
    
    
    verb = st.text_input("Enter verb here 👇 ... e.g., sell", "", max_chars=50)
    
    product = st.text_input("Enter product type here 👇 ... e.g., a product", "", max_chars=50)
      
    # run LLM generator 
    result = generate_text(disc_type, verb, product)
      
    st.write("LLM generated content: \n{}".format(result)   
      
      
      
      
      
      
      
    
    
