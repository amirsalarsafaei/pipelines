
"""
title: LangChain Pipeline
author: open-webui
date: 2024-05-30
version: 1.0
license: MIT
description: A pipeline for retrieving relevant information using LangChain with tool binding support.
requirements: langchain, langchain-openai, datasets>=2.6.1
"""

import json
import os
from typing import Generator, Iterator, List, Union

from git import Repo
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage, SystemMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import Tool
from langchain_community.document_loaders import GitLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel


class Pipeline:

    class Valves(BaseModel):
        """Options to change from the WebUI"""

        OPENAI_API_BASE_URL: str = "https://api.openai.com/v1"
        OPENAI_API_KEY: str = ""

    def __init__(self):

        self.id = "kenar-docs-helper"
        self.name = "Kenar Helper"

        self.valves = self.Valves()

        self.rag_chain = None
        self.tools = []
        self.retriever = None
        
    async def on_startup(self):
        # Initialize OpenAI chat model with tool binding capability
        self.llm = ChatOpenAI(
            temperature=0.7,
            model="gpt-4o",
            base_url=self.valves.OPENAI_API_BASE_URL,
            api_key=self.valves.OPENAI_API_KEY,
        )

        self._initialize_open_api_doc()
        self._initialize_github()
        self._initialize_vector_stores()

        # Create the base prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a specialized API integration assistant for Kenar, Divar Co.'s developer platform. Your role is to help developers understand and implement Kenar APIs in their applications.

You have access to Kenar's official documentation and API specifications. When answering:
- Provide clear, practical explanations of API endpoints, parameters, and responses
- Include relevant code examples when appropriate
- Explain authentication requirements and best practices
- Highlight any rate limits, restrictions, or important considerations
- Reference specific sections of the documentation when possible
- If a question is unclear or outside Kenar's capabilities, ask for clarification or explain the limitations

Remember: Your goal is to help developers successfully integrate with Kenar APIs, similar to how developers work with Shopify or Spotify APIs."""),
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
        with open('./kenar-apis.json', 'r') as f:
            self.open_api_doc = json.load(f)

    def _initialize_github(self):
        """Load and process documents from github.com/divar-ir/kenar-docs repository"""
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

        try:
            # Retrieve relevant context using the combined retriever
            relevant_docs = self.retriever(user_message)
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
            
            # Augment the user message with the retrieved context
            augmented_message = f"""Use the following context to help answer the question:

Context:
{context}

Question: {user_message}"""

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
                "input": augmented_message,
                "chat_history": chat_history
            })

            return response["output"]

        except Exception as e:
            return f"An error occurred: {str(e)}"
