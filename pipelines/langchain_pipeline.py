

from typing import Generator, Iterator, List, Union


class Pipeline:
    def __init__(self):
        self.rag_chain = None
        self.tools = []
        self.retriever = None
        
    async def on_startup(self):
        from langchain.agents import (AgentExecutor,
                                      create_openai_functions_agent)
        from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
        from langchain_openai import ChatOpenAI

        # Initialize OpenAI chat model with tool binding capability
        self.llm = ChatOpenAI(
            temperature=0.7,
            model="gpt-4o"
        
        )

        self._initialize_open_api_doc()
        self._initialize_github()
        self._initialize_vector_stores()

        # Create the base prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant with access to a knowledge base. 
            Use the provided context to answer questions accurately. 
            If you're not sure about something, say so."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])

        # Initialize the agent with tools (to be added later)
        self.agent = create_openai_functions_agent(
            llm=self.llm,
            prompt=prompt,
            tools=self.tools
        )

        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
        )


    async def on_shutdown(self):
        # Cleanup code if needed
        pass

    def _initialize_open_api_doc(self):
        import json
        import os
        
        with open('./kenar-apis.json', 'r') as f:
            self.open_api_doc = json.load(f)

    def _initialize_github(self):
        """Load and process documents from github.com/divar-ir/kenar-docs repository"""
        import os

        from git import Repo
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_community.document_loaders import GitLoader

        repo_path = "./kenar-docs"
        repo_url = "https://github.com/divar-ir/kenar-docs.git"

        # Clone the repository if it doesn't exist
        if not os.path.exists(repo_path):
            Repo.clone_from(repo_url, repo_path)

        # Initialize the GitLoader for markdown files
        loader = GitLoader(
            repo_path=repo_path,
            branch="main",
            file_filter=lambda file_path: file_path.endswith(".md")
        )

        # Load the documents
        self.github_docs = loader.load()

        # Initialize text splitter for markdown documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )

        # Split the documents
        self.github_splits = text_splitter.split_documents(self.github_docs)

    def _initialize_vector_stores(self):
        """Initialize two vector stores: one for GitHub docs and one for OpenAPI spec"""

        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_community.vectorstores import FAISS
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings

        # Initialize OpenAI embeddings
        embeddings = OpenAIEmbeddings()

        # Process GitHub markdown documents
        self.github_vectorstore = FAISS.from_documents(
            documents=self.github_splits,
            embedding=embeddings
        )

        # Process OpenAPI documentation directly from self.open_api_doc
        api_texts = []
        
        # Process paths
        paths = self.open_api_doc.get("paths", {})
        for path, methods in paths.items():
            for method, details in methods.items():
                text = f"""
                Endpoint: {method.upper()} {path}
                Summary: {details.get('summary', '')}
                Description: {details.get('description', '')}
                Parameters: {details.get('parameters', [])}
                Responses: {details.get('responses', {})}
                """
                api_texts.append(text)

            # Process components/schemas
            components = self.open_api_doc.get("components", {})
            schemas = components.get("schemas", {})
            for schema_name, schema in schemas.items():
                text = f"""
                Schema: {schema_name}
                Type: {schema.get('type', '')}
                Description: {schema.get('description', '')}
                Properties: {schema.get('properties', {})}
                """
                api_texts.append(text)

            # Create Document objects
            from langchain.schema import Document
            api_docs = [
                Document(
                    page_content=text,
                    metadata={"source": "openapi_spec"}
                )
                for text in api_texts
            ]

        # Split the API documentation
        api_text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n---\n", "\n\n", "\n", " "]
        )
        api_splits = api_text_splitter.split_documents(api_docs)

        # Create vector store for API documentation
        self.api_vectorstore = FAISS.from_documents(
            documents=api_splits,
            embedding=embeddings
        )

        # Create combined retriever
        self.retriever = self._create_combined_retriever()

    def _create_combined_retriever(self):
        """Create a combined retriever that searches both vector stores"""
        return lambda query: (
            self.github_vectorstore.similarity_search(query, k=2) +
            self.api_vectorstore.similarity_search(query, k=2)
        )

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        """
        Process a user message through the RAG pipeline with tool support.

        Args:
            user_message: The user's input message
            model_id: The model identifier
            messages: Previous conversation messages
            body: Additional request body parameters

        Returns:
            A response from the RAG chain or agent
        """
        from langchain.schema import HumanMessage, SystemMessage
        try:
            # Convert message history to the format expected by LangChain
            chat_history = []
            for msg in messages[:-1]:  # Exclude the current message
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "user":
                    chat_history.append(HumanMessage(content=content))
                elif role == "assistant":
                    chat_history.append(SystemMessage(content=content))

            response = self.agent_executor.invoke({
                "input": user_message,
                "chat_history": chat_history
            })

            return response["output"]

        except Exception as e:
            return f"An error occurred: {str(e)}"
