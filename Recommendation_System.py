import pandas as pd
import re
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_community.output_parsers.rail_parser import GuardrailsOutputParser
from typing import List, Dict

class RecommendationSystem:
    def __init__(self, chroma_db_path, llm_model='llama3'):
        self.local_llm = llm_model
        self.loader = CSVLoader(file_path=chroma_db_path)
        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=250, chunk_overlap=0)
        self.vectorstore = None
        self.retriever = None
        self.llm = None
        self.prompt_template_retrieval_grader = None
        self.retrieval_grader = None
        self.prompt_template_rag_chain = None
        self.rag_chain = None
        self.prompt_template_hallucination_grader = None
        self.hallucination_grader = None
        self.prompt_template_answer_grader = None
        self.answer_grader = None

        self._initialize_system()

    def _initialize_system(self):
        """Initializes the recommendation system by loading data, setting up vector store, retriever, and LLM."""
        try:
            # Load data
            data = self.loader.load()
            print("Data loaded successfully")
            
            # Split texts
            texts = self.text_splitter.split_documents(data)
            print(f"Texts split into {len(texts)} chunks")
            
            # Initialize embeddings and vector store
            self.vectorstore = Chroma.from_documents(
                documents=texts,
                collection_name="rag-chroma",
                embedding=HuggingFaceEmbeddings()
            )
            print("Vector store initialized")
            
            # Initialize retriever
            self.retriever = self.vectorstore.as_retriever()
            print("Retriever initialized")
            
        except Exception as e:
            print(f"Error initializing system: {e}")
            self.vectorstore = None
            self.retriever = None

        try:
            self.llm = Ollama(model=self.local_llm, format="json", temperature=0)

            self.prompt_template_retrieval_grader = PromptTemplate(
                template="""system You are a grader assessing relevance 
                of a retrieved document to a user question. If the document contains keywords related to the user question, 
                grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals.
                Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
                Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
                Here is the retrieved document:
                {documents}
                Here is the user question: {question}
                """,
                input_variables=["question", "documents"],
            )

            self.retrieval_grader = self.prompt_template_retrieval_grader | self.llm | JsonOutputParser()

            self.prompt_template_rag_chain = PromptTemplate(
                template="""system You are an AI tour guide named Ateema. Your role is to assist users in finding interesting places to visit. 
                            Greet the user warmly and provide detailed recommendations based on their specified location and interests.
                            Ensure the response includes exactly 2 recommendations for each of the following categories: dining, beverages, entertainment, 
                            cultural activities, outdoor activities, educational activities, and shopping based on {question}.
                            Recommendations in each category should be sequentially numbered, continuing from one category to the next, resulting in a total of 14 recommendations. Make sure to go with a title and a description 
                            Please avoid being a cut off generating recommendations. Make sure to conclude with a professional closing statement, making sure to not mention reaching out again.
                            Here is the user question: {question}
                            Here are some places you can recommend based on the retrieved documents:
                            {context}""",
                input_variables=["question", "context"],
            )

            self.rag_chain = self.prompt_template_rag_chain | self.llm | StrOutputParser()

            self.prompt_template_hallucination_grader = PromptTemplate(
                template="""system You are a grader assessing whether an answer is grounded in and supported by a set of facts from the provided documents. 
                            Your goal is to ensure the answer does not include any information that contradicts the documents. 
                            If the answer contains any information that directly contradicts the documents, grade it as 'no'. 
                            If the answer is generally consistent with the facts in the documents and does not introduce significant contradictions, grade it as 'yes'. 
                            Missing information is acceptable as long as the provided information aligns with the overall content of the documents.
                            Provide the binary score 'yes' or 'no' as a JSON with a single key 'score' and no preamble or explanation.

                            In addition to that above criteria, evaluate the following aspects:
                            1. **Consistency**: Does the answer generally align with the facts provided in the documents?
                            2. **Completeness**: While the answer may not cover all details, it should not include any incorrect or invented details.
                            3. **Relevance**: The answer should be relevant to the question and broadly based on the facts from the documents.
                            
                            Here are the facts from the documents:
                            {documents}
                            
                            Here is the generated answer:
                            {generation}""",
                input_variables=["generation", "documents"]
            )

            self.hallucination_grader = self.prompt_template_hallucination_grader | self.llm | JsonOutputParser()

            self.prompt_template_answer_grader = PromptTemplate(
                template="""system You are a grader assessing whether an 
                answer is useful to resolve a question. Give a binary score 'yes' or 'no' to indicate whether the answer is 
                useful to resolve a question. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
                Here is the answer:
                {generation}
                Here is the question: {question}""",
                input_variables=["generation", "question"],
            )

            self.answer_grader = self.prompt_template_answer_grader | self.llm | JsonOutputParser()
            
        except Exception as e:
            print(f"Error initializing system: {e}")

    def construct_question(self, customer_info: Dict) -> str:
        """Constructs a question string based on customer information."""
        try:
            first_name = customer_info.get("first_name", "").strip().split()[0]
            travel_city = customer_info.get("travel_city")
            start_date = customer_info.get("start_date")
            end_date = customer_info.get("end_date")
            group_size = customer_info.get("group_size")
            group_age = customer_info.get("group_age")
            pref_dining = customer_info.get("pref_dining")
            pref_beverage = customer_info.get("pref_beverage")
            pref_entertainment = customer_info.get("pref_entertainment")
            pref_cultural = customer_info.get("pref_cultural")
            pref_outdoor = customer_info.get("pref_outdoor")
            pref_education = customer_info.get("pref_education")
            pref_shop = customer_info.get("pref_shop")

            question = (f"My name is {first_name}. I will be visiting {travel_city} from {start_date} to {end_date}. "
                        f"I will be traveling with a group of {group_size} people, aged {group_age}. "
                        f"Our preferences include dining ({pref_dining}), beverages ({pref_beverage}), entertainment ({pref_entertainment}), "
                        f"cultural activities ({pref_cultural}), outdoor activities ({pref_outdoor}), educational activities ({pref_education}), "
                        f"and shopping ({pref_shop}). Can you recommend some interesting places to visit?")
            
            return question
        except Exception as e:
            print(f"Error constructing question: {e}")
            return ""

    def retrieve(self, state: Dict) -> Dict:
        """Retrieves relevant documents based on the question."""
        try:
            print("---RETRIEVE---")
            question = state["question"]
            documents = self.retriever.invoke(question)
            return {"documents": documents, "question": question}
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return {}
        
    def generate(self, state: Dict) -> Dict:
        try:
            print("---GENERATE---")
            question = state["question"]
            documents = state["documents"]
            context = "\n\n".join([doc.page_content for doc in documents])
            
            prompt = self.prompt_template_rag_chain.format(question=question, context=context)

            generation = ""
            attempts = 0
            max_attempts = 3
            
            while attempts < max_attempts:
                generation = self.llm.invoke(prompt)
                parsed_generation = StrOutputParser().parse(generation)
                attempts += 1
                
                print(f"Generated response attempt {attempts}: {parsed_generation}")

                # Check if the response is complete
                if self.is_complete(parsed_generation):
                    return {"documents": documents, "question": question, "generation": parsed_generation.strip()}  # Added strip() method
                else:
                    print("Incomplete response, retrying...")

            print(f"Incomplete response after {max_attempts} attempts")
            parsed_generation = "Sorry, I couldn't generate a complete response. Please try again."
            return {"documents": documents, "question": question, "generation": parsed_generation.strip()}  # Added strip() method
        except Exception as e:
            print(f"Error generating response: {e}")
            return {"documents": documents, "question": question, "generation": ""}

    def generate_recommendation(self, customer_info: Dict) -> Dict:
        """Generates recommendations based on customer information."""
        try:
            question = self.construct_question(customer_info)
            state = {"question": question}

            state = self.route_question(state)
            state = self.retrieve(state)
            state = self.grade_documents(state)
            decision = self.decide_to_generate(state)
            
            iteration_count = 0
            max_iterations = 3  # Adjusted to allow a reasonable number of iterations
            
            while decision != "useful" and iteration_count < max_iterations:
                iteration_count += 1
                print(f"Iteration {iteration_count} : Generating response")
                state = self.generate(state)
                
                # Print the generated response before hallucination check
                print(f"Generated response: {state.get('generation', 'No generation')}")

                decision = self.grade_generation_v_documents_and_question(state)
                print(f"Grading decision: {decision}")

                if decision == "not supported":
                    state = self.retrieve(state)  # Retrieve new documents if the generation is not supported

            final_response = self.parse_response(state.get("generation", ""), state.get("documents", []))
            return final_response
        except Exception as e:
            print(f"Error generating recommendation: {e}")
            return {}

    def is_complete(self, response: str) -> bool:
        """Checks if the generated response is complete."""
        required_sections = [
            "dining",
            "beverages",
            "entertainment",
            "cultural activities",
            "outdoor activities",
            "educational activities",
            "shopping"
        ]
        response_lower = response.lower()
        return all(section in response_lower for section in required_sections)

    def grade_documents(self, state: Dict) -> Dict:
        """Grades the relevance of retrieved documents to the question."""
        try:
            print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
            question = state["question"]
            documents = state["documents"]
            filtered_docs = []
            for d in documents:
                score = self.retrieval_grader.invoke({"question": question, "documents": d.page_content})
                grade = score['score']
                if grade.lower() == "yes":
                    print("---GRADE: DOCUMENT RELEVANT---")
                    filtered_docs.append(d)
                else:
                    print("---GRADE: DOCUMENT NOT RELEVANT---")
            return {"documents": filtered_docs, "question": question}
        except Exception as e:
            print(f"Error grading documents: {e}")
            return {"documents": [], "question": state["question"]}

    def route_question(self, state: Dict) -> Dict:
        print("---ROUTE QUESTION---")
        question = state["question"]
        print(question)
        return state

    def decide_to_generate(self, state: Dict) -> str:
        """Decides whether to generate a response based on graded documents."""
        print("---ASSESS GRADED DOCUMENTS---")
        filtered_documents = state["documents"]
        if filtered_documents:
            print("---DECISION: GENERATE---")
            return "generate"
        else:
            print("---DECISION: NOT USEFUL---")
            return "not useful"

    def grade_generation_v_documents_and_question(self, state: Dict) -> str:
        print("---CHECK HALLUCINATIONS---")
        documents = state["documents"]
        generation = state["generation"]
        score = self.hallucination_grader.invoke({"documents": documents, "generation": generation})
        grade = score['score']
        if grade == "yes":
            print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
            score = self.answer_grader.invoke({"question": state["question"], "generation": generation})
            grade = score['score']
            if grade == "yes":
                print("---DECISION: GENERATION ADDRESSES QUESTION---")
                return "useful"
            else:
                print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                return "not useful"
        else:
            print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
            return "not supported"
        
    def parse_response(self, response: str, documents: List[Document]) -> Dict:
        """Parses the generated response and matches it with the original CSV data to extract URLs."""
        try:
            # Extract titles with numbers
            titles_with_numbers = re.findall(r'\d+\.\s([^-\n]+)', response)

            normalized_titles = [self._normalize_text(title) for title in titles_with_numbers]

            # Load the original CSV file to match against the title_cleaned and get the image URLs
            df = pd.read_csv(self.loader.file_path)
            df['normalized_title_cleaned'] = df['title_cleaned'].apply(self._normalize_text)

            recommendations = []
            url_info = []
            for index, (title_with_number, normalized_title) in enumerate(zip(titles_with_numbers, normalized_titles), start=1):
                match = df[df['normalized_title_cleaned'].str.contains(normalized_title, na=False, case=False)]
                print(f"Matching '{normalized_title}' with titles in CSV, found matches:", match)
                
                if not match.empty:
                    url = match.iloc[0]['image']
                    match_status = "Yes"
                    url_status = url if pd.notna(url) else "No URL available"
                else:
                    url_status = "No URL available"
                    match_status = "No"
                recommendations.append(f"{index}. {title_with_number.strip()}")
                url_info.append({"Title": title_with_number.strip(), "Match": match_status, "URL": url_status})

            # Create a human-readable formatted response
            formatted_prompt = response.strip().replace("\\n", "\n").replace("\\'", "'")

            formatted_response = {
                "recommendation": formatted_prompt,
                "url_information": url_info
            }

            return formatted_response
        except Exception as e:
            print(f"Error parsing response: {e}")
            return {"recommendation": "", "url_information": []}

    def _normalize_text(self, text: str) -> str:
        """Normalizes text for matching purposes."""
        return re.sub(r'\s+', '', text.lower().replace("the", "").replace("'", "").replace("-", "").replace(",", "").replace(".", ""))
