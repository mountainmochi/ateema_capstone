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
import textwrap

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
        self.prompt_template_bias_grader = None 
        self.bias_grader = None  
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
            self.llm = Ollama(model=self.local_llm, format="json", temperature=0.2)

            self.prompt_template_retrieval_grader = PromptTemplate(
                template="""system You are a grader assessing relevance
                of a retrieved document to a user question. If the document contains keywords related to the user question,
                grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals.
                Give a binary score 'pass' or 'fail' score to indicate whether the document is relevant to the question.
                Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
                Here is the retrieved document:
                {documents}
                Here is the user question: {question}
                """,
                input_variables=["question", "documents"],
            )

            self.retrieval_grader = self.prompt_template_retrieval_grader | self.llm | JsonOutputParser()

            self.prompt_template_rag_chain = PromptTemplate(
                template="""system: You are Ateema, an AI tour guide. Respond to the user's query using the following format precisely:

                            1. Start with "Hello [user's first name]! I'm Ateema, your AI tour guide."

                            2. Acknowledge the user's destination and group details if mentioned.

                            3. Provide exactly 2 recommendations for each of these categories: Dining, Beverages, Entertainment, Cultural Activities, Outdoor Activities, Educational Activities, and Shopping.

                            4. Format each category and recommendation like below:

                              **Category Name:**
                              1. **Place Name**: Brief description of the place and why it's suitable for the group.
                              2. **Place Name**: Brief description of the place and why it's suitable for the group.

                            5. End with a brief closing statement wishing them a good trip, without inviting further questions.

                            6. End with "I hope you enjoy your trip to Chicago! Have a great time exploring this amazing city."

                            7. Do not include any disclaimers or apologies about the information provided.

                            8. Ensure the entire response is one continuous paragraph without line breaks between sections.

                            9. Bold each recommendation place

                            User question: {question}

                            Context for recommendations: {context}""",
                input_variables=["question", "context"],
            )

            self.rag_chain = self.prompt_template_rag_chain | self.llm | StrOutputParser()


            self.prompt_template_bias_grader = PromptTemplate(
                template="""system You are a grader assessing whether the answer contains any biased information.
                            The goal is to ensure the answer is neutral, objective, and fair.
                            If the answer contains any biased statements, stereotypes, or shows preference towards a particular group or viewpoint, grade it as 'fail'.
                            If the answer is neutral, does not show any bias, and is inclusive, grade it as 'pass'.
                            Provide the binary score 'pass' or 'fail' as a JSON with a single key 'score' and no preamble or explanation.

                            In addition to the above criteria, evaluate the following aspects:
                            1. **Neutrality**: Does the answer avoid biased language or favoritism towards any group or viewpoint?
                            2. **Inclusivity**: Is the answer inclusive and respectful of diversity, avoiding stereotypes?
                            3. **Fairness**: Does the answer provide a balanced perspective without promoting one side over another?

                            Here is the generated answer:
                            {generation}""",
                input_variables=["generation"]
            )

            self.bias_grader = self.prompt_template_bias_grader | self.llm | JsonOutputParser()

            self.prompt_template_hallucination_grader = PromptTemplate(
                template="""system You are a grader assessing whether an answer is grounded in and supported by a set of facts from the provided documents.
                            Your goal is to ensure the answer does not include any information that contradicts the documents.
                            If the answer contains any information that directly contradicts the documents, grade it as 'fail'.
                            If the answer is generally consistent with the facts in the documents and does not introduce significant contradictions, grade it as 'pass'.
                            Missing information is acceptable as long as the provided information aligns with the overall content of the documents.
                            Provide the binary score 'pass' or 'fail' as a JSON with a single key 'score' and no preamble or explanation.

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
                template="""You are a grader assessing whether an answer about a recommended Chicago place is useful and informative.
                            Give a binary score 'pass' or 'fail' to indicate whether the answer relates to Chicago.
                            Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
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

                # Check if the response is complete and properly formatted
                if self.is_complete_and_formatted(parsed_generation):
                    return {"documents": documents, "question": question, "generation": parsed_generation.strip()}
                else:
                    print("Incomplete or improperly formatted response, retrying...")

            print(f"Incomplete response after {max_attempts} attempts")
            parsed_generation = "Sorry, I couldn't generate a complete response. Please try again."
            return {"documents": documents, "question": question, "generation": parsed_generation.strip()}  # Added strip() method
        except Exception as e:
            print(f"Error generating response: {e}")
            return {"documents": documents, "question": question, "generation": ""}

    def generate_recommendation(self, customer_info: Dict) -> Dict:
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


    def is_complete_and_formatted(self, response: str) -> bool:
        """Checks if the generated response is complete and properly formatted."""
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

        # Check if all required sections are present
        missing_sections = [section for section in required_sections if section not in response_lower]
        if missing_sections:
            print(f"Format check failed: Missing sections - {', '.join(missing_sections)}")
            return False

        # Check if each section is properly formatted
        section_pattern = r"\*\*(.+?):\*\*\s+1\. \*\*(.+?)\*\*: .+?\s+2\. \*\*(.+?)\*\*: .+?"
        sections = re.findall(section_pattern, response, re.DOTALL)

        if len(sections) != len(required_sections):
            print(f"Format check failed: Expected {len(required_sections)} formatted sections, found {len(sections)}")
            return False

        # Check if all required sections are present in the correct format
        formatted_sections = [section[0].lower() for section in sections]
        missing_formatted_sections = [section for section in required_sections if section not in formatted_sections]
        if missing_formatted_sections:
            print(f"Format check failed: Missing or incorrectly formatted sections - {', '.join(missing_formatted_sections)}")
            return False

        # Check if the response ends with a closing statement
        # if not re.search(r"I hope you enjoy your trip to Chicago! Have a great time exploring this amazing city\.$", response):
        #     print("Format check failed: Missing or incorrect closing statement")
        #     return False

        print("Format check passed: All criteria met")
        return True

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
                if grade.lower() == "pass":
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
        print("---CHECK BIASES---")
        generation = state["generation"]
        bias_score = self.bias_grader.invoke({"generation": generation})
        grade = bias_score['score']
        if grade == "pass":
            print("---DECISION: GENERATION IS NOT BIASED---")

            print("---CHECK HALLUCINATIONS---")
            documents = state["documents"]
            hallucination_score = self.hallucination_grader.invoke({"documents": documents, "generation": generation})
            grade = hallucination_score['score']
            if grade == "pass":
                print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
                answer_score = self.answer_grader.invoke({"question": state["question"], "generation": generation})
                grade = answer_score['score']
                if grade == "pass":
                    print("---DECISION: GENERATION ADDRESSES QUESTION---")
                    return "useful"
                else:
                    print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                    return "not useful"
            else:
                print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
                return "not supported"
        else:
            print("---DECISION: GENERATION IS BIASED---")
            return "not useful"


    def parse_response(self, response: str, documents: List[Document]) -> Dict:
        """Parses the generated response and matches it with the original CSV data to extract URLs."""
        try:
            # Remove any JSON-like wrapping if present
            response = re.sub(r'^.*?"value":\s*"', '', response)
            response = re.sub(r'"}\s*$', '', response)

            # Unescape any escaped characters
            response = response.replace('\\r\\n', '\n').replace('\\"', '"')

            # Extract titles with numbers
            titles_with_numbers = re.findall(r'\d+\.\s*\*\*(.*?)\*\*:', response, re.DOTALL)

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
            formatted_prompt = response.strip()

            formatted_response = {
                "recommendation": formatted_prompt,
                "url_information": url_info
            }

            return formatted_response
        except Exception as e:
            print(f"Error parsing response: {e}")
            return {"recommendation": response, "url_information": []}

    def _normalize_text(self, text: str) -> str:
        """Normalizes text for matching purposes."""
        return re.sub(r'\s+', '', text.lower().replace("the", "").replace("'", "").replace("-", "").replace(",", "").replace(".", ""))
