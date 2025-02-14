"""
title: Kenar Docs Pipeline (Haystack Version)
description: A pipeline for handling Kenar API documentation and chat interactions using Haystack
author: divar-ir
author_url: https://github.com/divar-ir
funding_url: https://github.com/divar-ir/kenar-docs
version: 0.3
"""

import json
import logging
import os
import re
from copy import deepcopy
from typing import Any, Dict, Generator, Iterator, List, Optional, Tuple, Union

from fastapi import HTTPException
from git import Repo
from haystack import Document, Pipeline, component
from haystack.components.embedders import OpenAIDocumentEmbedder
from haystack.components.generators import OpenAIGenerator
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential


class Pipe:
    """
    A Haystack-based pipeline for handling Kenar API documentation and chat interactions.
    This pipe provides a natural language interface to Kenar's API documentation,
    allowing users to ask questions and get relevant information about the APIs.
    """

    class Valves(BaseModel):
        """Configuration options for the Kenar Docs Pipeline"""

        OPENAI_API_BASE: str = Field(
            default="",
            description="Base URL for OpenAI API",
        )
        OPENAI_API_KEY: str = Field(
            default="",
            description="OpenAI API key for the pipeline",
        )

    class UserValves(BaseModel):
        """User-specific configuration for Kenar API access"""

        KENAR_API_KEY: str = Field(
            default="", description="User's Kenar API key for authentication"
        )
        OAUTH_CLIENT_SECRET: str = Field(
            default="", description="OAuth client secret for authentication"
        )
        ACCESS_TOKEN: str = Field(
            default="", description="Access token for API authentication"
        )

    def __init__(self):
        """
        Initialize the Kenar Docs Pipeline using Haystack components.
        Sets up the document store, embedder, retriever, and generator.
        """
        # Pipeline identification
        self.id = "kenar-docs-3"
        self.name = "Kenar Documentation Assistant"
        self.description = "Natural language interface to Kenar API documentation"

        # Initialize logger
        self.logger = self._setup_logger()
        self.logger.info("Initializing Haystack pipeline components...")

        # Initialize valves with environment variables and validation
        try:
            self.valves = self.Valves()
            self.user_valves = self.UserValves()
        except ValueError as e:
            raise HTTPException(
                status_code=500, detail=f"Invalid pipeline configuration: {str(e)}"
            )

        # Set OpenAI environment variables
        os.environ["OPENAI_API_KEY"] = self.valves.OPENAI_API_KEY
        os.environ["OPENAI_API_BASE"] = self.valves.OPENAI_API_BASE

        # Initialize Haystack components
        self.document_store = InMemoryDocumentStore()
        self.logger.info("document_store initializaed")

        self.retriever = InMemoryEmbeddingRetriever(
            document_store=self.document_store, top_k=5
        )
        self.logger.info("retriever initializaed")

        self.embedder = OpenAIDocumentEmbedder(
            # api_key=self.valves.OPENAI_API_KEY,
            model="text-embedding-3-large",
            api_base_url=self.valves.OPENAI_API_BASE,
        )
        self.logger.info("embedder initializaed")

        self.generator = OpenAIGenerator(
            api_base_url=self.valves.OPENAI_API_BASE,
            model="gpt-4o",
            generation_kwargs={
                "temperature": 0.3,
            },
        )

        # Create prompt builder component
        @component
        class PromptBuilder:
            @component.output_types(prompt=str)
            def run(self, documents: List[Document], query: str):
                context = "\n".join([doc.content for doc in documents])
                prompt = f"""Context: {context}\n\nQuestion: {query}\n\nAnswer:"""
                return {"prompt": prompt}

        self.prompt_builder = PromptBuilder()

        # Initialize the main RAG pipeline
        self.rag_pipeline = Pipeline()
        self.rag_pipeline.add_component("retriever", self.retriever)
        self.rag_pipeline.add_component("prompt_builder", self.prompt_builder)
        self.rag_pipeline.add_component("generator", self.generator)

        # Connect components
        self.rag_pipeline.connect("retriever", "prompt_builder")
        self.rag_pipeline.connect("prompt_builder", "generator")

        # Initialize documents and API components
        self._initialize_open_api_doc()
        self._initialize_github()
        self._initialize_document_store()

    def _setup_logger(self) -> logging.Logger:
        """Set up a logger for the pipeline"""
        logger = logging.getLogger("kenar_pipeline")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def _initialize_open_api_doc(self):
        """Initialize and load the OpenAPI documentation"""
        try:
            api_spec_path = os.path.join(
                os.path.dirname(__file__), "..", "kenar-apis.json"
            )
            with open(api_spec_path, "r") as f:
                self.open_api_doc = json.load(f)

            # Convert OpenAPI spec to documents
            api_docs = []
            for path, path_data in self.open_api_doc["paths"].items():
                for method, operation in path_data.items():
                    content = f"""
                    Path: {path}
                    Method: {method.upper()}
                    Summary: {operation.get('summary', '')}
                    Description: {operation.get('description', '')}
                    Parameters: {json.dumps(operation.get('parameters', []), indent=2)}
                    """
                    api_docs.append(
                        Document(
                            content=content,
                            meta={
                                "path": path,
                                "method": method,
                                "type": "api_doc",
                                "operation_id": operation.get("operationId", ""),
                            },
                        )
                    )

            self.api_docs = api_docs
            self.logger.info(f"Loaded {len(api_docs)} API endpoints from specification")

        except FileNotFoundError:
            self.logger.error("OpenAPI specification file not found")
            raise HTTPException(
                status_code=500, detail="OpenAPI specification file not found"
            )
        except json.JSONDecodeError:
            self.logger.error("Invalid OpenAPI specification format")
            raise HTTPException(
                status_code=500, detail="Invalid OpenAPI specification format"
            )

    def _initialize_github(self):
        """Initialize and load GitHub documentation"""
        try:
            # Clone or update the repository
            repo_path = os.path.join(os.path.dirname(__file__), "..", "kenar-docs")
            if not os.path.exists(repo_path):
                self.logger.info("Cloning documentation repository...")
                Repo.clone_from("https://github.com/divar-ir/kenar-docs.git", repo_path)
            else:
                self.logger.info("Pulling latest documentation...")
                repo = Repo(repo_path)
                repo.remotes.origin.pull()

            # Process markdown files
            github_docs = []
            for root, _, files in os.walk(repo_path):
                for file in files:
                    if file.endswith(".md"):
                        file_path = os.path.join(root, file)
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()
                            github_docs.append(
                                Document(
                                    content=content,
                                    meta={
                                        "source": "github",
                                        "file_path": file_path,
                                        "type": "markdown",
                                    },
                                )
                            )

            self.github_docs = github_docs
            self.logger.info(
                f"Loaded {len(github_docs)} documentation files from GitHub"
            )

        except Exception as e:
            self.logger.error(f"Failed to initialize GitHub documentation: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to initialize GitHub documentation: {str(e)}",
            )

    def _initialize_document_store(self):
        """Initialize the document store with API and GitHub documentation"""
        try:
            # Combine all documents
            all_docs = self.api_docs + self.github_docs

            # Embed and index documents
            self.logger.info("Embedding and indexing documents...")
            # Embed documents using OpenAIDocumentEmbedder
            documents_with_embeddings = self.embedder.run(all_docs)["documents"]
            self.document_store.write_documents(documents_with_embeddings)

            self.logger.info(f"Successfully indexed {len(all_docs)} documents")

        except Exception as e:
            self.logger.error(f"Failed to initialize document store: {e}")
            raise HTTPException(
                status_code=500, detail=f"Failed to initialize document store: {str(e)}"
            )

    def _create_prompt_template(self) -> str:
        """Create the prompt template for the generator"""
        return """شما دستیار هوشمند API کنار هستید. همیشه باید به زبان فارسی پاسخ دهید و اطلاعات را از مستندات OpenAPI در اولویت قرار دهید. هدف شما کمک به توسعه‌دهندگان برای درک و استفاده از APIهای کنار است.

Context: {context}

Chat History: {chat_history}

User Question: {query}

برای هر پاسخ، این موارد را رعایت کنید:

۱. شروع با جزئیات endpoint مربوطه:
   - متد HTTP و مسیر
   - پارامترهای ضروری و نوع آنها
   - ساختار درخواست و پاسخ

۲. توضیح کاربرد API در یک مثال کاربردی واقعی

۳. ارائه مثال کامل curl در صورت نیاز

۴. ارائه نمونه کد در زبان‌های برنامه‌نویسی رایج در صورت درخواست

۵. توضیح نکات مهم و محدودیت‌های API
"""

    def _format_chat_history(self, messages: List[Dict[str, str]]) -> str:
        """Format chat history for the prompt template"""
        formatted_history = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role in ["user", "assistant"]:
                formatted_history.append(f"{role.title()}: {content}")
        return "\n".join(formatted_history)

    def pipe(self, body: Dict[str, Any], *args, **kwargs) -> Union[str, Dict[str, Any]]:
        """
        Process a user message through the Haystack pipeline.

        Args:
            body: Request body containing messages and other parameters

        Returns:
            Response from the pipeline
        """
        try:
            messages = body.get("messages", [])
            user_message = next(
                (
                    msg.get("content", "")
                    for msg in reversed(messages)
                    if msg.get("role") == "user"
                ),
                None,
            )

            if not user_message:
                raise HTTPException(
                    status_code=400, detail="No valid user message found"
                )

            # Detect language and request intent
            is_request, is_persian = self._detect_request_intent(user_message)

            # Get relevant documents
            retriever_output = self.retriever.run(
                query=user_message, documents=self.document_store.filter_documents()
            )

            # Format context from retrieved documents
            context = "\n".join(
                [
                    f"[Source: {doc.meta.get('type', 'unknown')}]\n{doc.content}"
                    for doc in retriever_output["documents"]
                ]
            )

            # Format chat history
            chat_history = self._format_chat_history(messages[:-1])

            # Run the RAG pipeline
            pipeline_output = self.rag_pipeline.run(query=user_message)
            generator_output = pipeline_output["generator"]

            return {"response": generator_output["replies"][0]}

        except Exception as e:
            self.logger.error(f"Error in pipeline: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

    def inlet(
        self, body: Dict[str, Any], __user__: Optional[dict] = None
    ) -> Dict[str, Any]:
        """Validate incoming requests"""
        if not isinstance(body, dict):
            raise HTTPException(
                status_code=400, detail="Request body must be a dictionary"
            )

        if "messages" not in body:
            raise HTTPException(
                status_code=400, detail="Request body must contain 'messages' field"
            )

        return body

    def outlet(
        self, response: Dict[str, Any], __user__: Optional[dict] = None
    ) -> Dict[str, Any]:
        """Process and validate outgoing responses"""
        if not isinstance(response, dict):
            response = {"response": str(response)}

        return response

    def pipes(self) -> List[Dict[str, str]]:
        """Return available pipeline configurations"""
        return [{"id": self.id, "name": self.name, "description": self.description}]

    def _get_operation_details(
        self, path: str, method: str
    ) -> Optional[Dict[str, Any]]:
        """Get detailed API operation information"""
        try:
            operation = self.open_api_doc["paths"][path][method.lower()]
            return {
                "path": path,
                "method": method,
                "summary": operation.get("summary", ""),
                "description": operation.get("description", ""),
                "parameters": operation.get("parameters", []),
                "requestBody": operation.get("requestBody", {}),
                "responses": operation.get("responses", {}),
            }
        except KeyError:
            return None
