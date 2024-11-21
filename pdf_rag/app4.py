import os
from typing import List
from operator import itemgetter

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


class PDFMultipleChoiceGenerator:
    def __init__(self, data_dir: str, openai_api_key: str, db_path: str = "faiss_index"):
        self.data_dir = data_dir
        self.db_path = db_path
        self.openai_api_key = openai_api_key
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.vector_store = self.load_or_create_vector_store()
        self.chain = self.setup_chain()

    def load_documents(self) -> List:
        print("Loading PDF files...")
        documents = []
        for file in os.listdir(self.data_dir):
            if file.endswith(".pdf"):
                file_path = os.path.join(self.data_dir, file)
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
        return documents

    def load_or_create_vector_store(self):
        if os.path.exists(self.db_path):
            print("Loading existing vector store...")
            return FAISS.load_local(self.db_path, self.embeddings, allow_dangerous_deserialization=True)
        else:
            print("Creating new vector store...")
            documents = self.load_documents()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = text_splitter.split_documents(documents)
            vector_store = FAISS.from_documents(texts, self.embeddings)
            vector_store.save_local(self.db_path)
            return vector_store

    def setup_chain(self):
        # Initialize ChatOpenAI with updated configuration
        llm = ChatOpenAI(
            temperature=0.7,
            model="gpt-4",
            openai_api_key=self.openai_api_key,  # Pass API key directly
            seed=42,
            streaming=True
        )
        
        template = """You are tasked with creating multiple-choice questions based EXCLUSIVELY on the following context retrieved from a FAISS database. This database contains information extracted from PDF documents about AI, machine learning, and related technologies. Your task is to generate questions ONLY from this provided context, without adding any external information or knowledge.

Context from FAISS database:
{context}

Instructions:
1. Create EXACTLY 10 multiple-choice questions based SOLELY on the information in the above context.
2. Each question must have 4 options (A, B, C, D) with only one correct answer.
3. Ensure that all questions and answers are derived directly from the given context.
4. Do NOT include any information that is not explicitly stated in the context.
5. Focus on technical details, parameters, use cases, and comparisons between different AI tools and concepts mentioned in the context.
6. If there isn't enough information for 10 unique questions, focus on different aspects or implications of the same information.

Here are some examples of the types of questions you should aim to create (but remember, only create questions based on the actual content in the given context):

1. What is top_p and what happens if you change it?
A) It's a hyperparameter that controls output randomness
B) It's a metric for model performance
C) It's a type of neural network architecture
D) It's a data preprocessing technique
Correct Answer: A

2. What is the purpose of LangChain?
A) To train language models
B) To create chain reactions in particle physics
C) To build applications with large language models
D) To translate between programming languages
Correct Answer: C

3. Why would you use the Vision API instead of DALL-E?
A) Vision API is faster
B) Vision API is for image analysis, while DALL-E is for image generation
C) Vision API is open-source
D) Vision API can generate higher quality images
Correct Answer: B

4. Can DALL-E generate multiple images at the same time?
A) No, it can only generate one image at a time
B) Yes, it can generate up to 10 images simultaneously
C) Yes, but only if they are related to the same prompt
D) No, DALL-E is for text generation, not image generation
Correct Answer: B

5. What is LCEL?
A) A type of neural network
B) A programming language for AI
C) LangChain Expression Language
D) A dataset for language modeling
Correct Answer: C

Now, generate 10 multiple-choice questions based on the provided context, following the format of these examples. Remember to only use information present in the context.

Format your response as follows:
1. Question
A) Option A
B) Option B
C) Option C
D) Option D
Correct Answer: [Letter]

2. Question
...

(Continue until you have 10 questions)

Remember: It is crucial that you generate EXACTLY 10 questions. All information must come strictly from the provided context."""

        # Create the prompt template
        prompt = ChatPromptTemplate.from_template(template)

        # Create the chain with improved error handling
        chain = (
            {"context": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        return chain

    def generate_questions(self) -> str:
        print("Generating multiple-choice questions...")
        try:
            # Get all the chunks from the vector store
            chunks = list(self.vector_store.docstore._dict.values())
            # Randomly shuffle and select chunks to fit within the context window
            import random
            random.shuffle(chunks)
            context = ""
            max_context_length = 3000  # Adjust based on model's context window
            for chunk in chunks:
                if len(context) + len(chunk.page_content) > max_context_length:
                    break
                context += chunk.page_content + "\n\n"
            
            # Use the chain with error handling
            response = self.chain.invoke(context)
            return response
        except Exception as e:
            print(f"Error generating questions: {str(e)}")
            return "An error occurred while generating questions. Please try again."

    def update_vector_store(self):
        print("Updating vector store...")
        try:
            new_documents = self.load_documents()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = text_splitter.split_documents(new_documents)
            self.vector_store.add_documents(texts)
            self.vector_store.save_local(self.db_path)
            print("Vector store updated and saved.")
        except Exception as e:
            print(f"Error updating vector store: {str(e)}")


def main():
    try:
        import config  # Make sure to create a config.py file with your API key
        
        data_dir = "data"
        openai_api_key = config.OPENAI_API_KEY
        db_path = "faiss_index"
        
        mcq_generator = PDFMultipleChoiceGenerator(data_dir, openai_api_key, db_path)
        
        print("Multiple Choice Question Generator")
        print("Type 'generate' to create questions, 'update' to refresh the vector store, or 'quit' to exit.")
        
        while True:
            user_input = input("\nEnter command (generate/update/quit): ").lower()
            
            if user_input == 'quit':
                break
            elif user_input == 'update':
                mcq_generator.update_vector_store()
            elif user_input == 'generate':
                questions = mcq_generator.generate_questions()
                print("\nGenerated Multiple Choice Questions:")
                print(questions)
            else:
                print("Invalid command. Please enter 'generate', 'update', or 'quit'.")

        print("Thank you for using the Multiple Choice Question Generator!")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Please make sure all required packages are installed and configured correctly.")


if __name__ == "__main__":
    main()