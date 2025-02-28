

from typing import Generator, Iterator, List, Union

from pydantic import BaseModel


class Pipeline:

    class Valves(BaseModel):
        OPENAI_API_BASE_URL: str = "https://api.openai.com/v1"
        OPENAI_API_KEY: str = ""
        pass

    def __init__(self):
        import os
        self.id = "kenar-docs"
        self.name = "Kenar Docs"
        self.rag_chain = None
        self.tools = []
        self.retriever = None

        self.valves = self.Valves(
            **{
                "OPENAI_API_KEY": os.getenv(
                    "OPENAI_API_KEY", "your-openai-api-key-here"
                ),
                "OPENAI_API_BASE_URL": os.getenv(
                    "OPENAI_API_BASE_URL", "https://api.openai.com/v1"
                ),

            },

        )
        
    def _get_operation_spec(self, operation_id: str) -> str:
        """
        Get the OpenAPI specification for a specific operation ID.
        
        Args:
            operation_id: The operation ID to look up
            
        Returns:
            A formatted string containing the operation specification
        """
        # Search through all paths and methods
        for path, methods in self.open_api_doc.get("paths", {}).items():
            for method, details in methods.items():
                if details.get("operationId") == operation_id:
                    # Format the response
                    parameters = details.get("parameters", [])
                    param_str = "\n".join([
                        f"- {p.get('name')} ({p.get('in')}): {p.get('description', 'No description')}"
                        for p in parameters
                    ])
                    
                    responses = details.get("responses", {})
                    response_str = "\n".join([
                        f"- {code}: {resp.get('description', 'No description')}"
                        for code, resp in responses.items()
                    ])
                    
                    return f"""Operation: {operation_id}
HTTP Method: {method.upper()}
Path: {path}
Summary: {details.get('summary', 'No summary')}
Description: {details.get('description', 'No description')}

Parameters:
{param_str}

Responses:
{response_str}
"""
        return f"No operation found with ID: {operation_id}"

    async def on_startup(self):
        from langchain.agents import (AgentExecutor,
                                      create_openai_functions_agent)
        from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
        from langchain.tools import Tool
        from langchain_openai import ChatOpenAI

        # Initialize OpenAI chat model with tool binding capability
        self.llm = ChatOpenAI(
            temperature=0.3,
            model="gpt-4o",
            api_key=self.valves.OPENAI_API_KEY,
            base_url=self.valves.OPENAI_API_BASE_URL,
        )

        self._initialize_open_api_doc()
        self._initialize_github()
        self._initialize_vector_stores()

        # Create the base prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", """شما دستیار هوشمند API کنار هستید. همیشه باید به زبان فارسی پاسخ دهید و اطلاعات را از مستندات OpenAPI در اولویت قرار دهید. هدف شما کمک به توسعه‌دهندگان برای درک و استفاده از APIهای کنار است.

برای هر پاسخ، این موارد را رعایت کنید:

۱. شروع با جزئیات endpoint مربوطه:
   - متد HTTP و مسیر
   - پارامترهای ضروری و نوع آنها
   - ساختار درخواست و پاسخ

۲. توضیح کاربرد API در یک مثال کاربردی واقعی

۳. ارائه مثال کامل curl:

   ```bash
   curl -X METHOD https://api.divar.ir/v1/open-platform/PATH \
     -H "X-API-Key: YOUR_API_KEY" \
     -H "X-Access-Token: OAUTH_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{{"param": "value"}}'
   ```

۴. ارائه نمونه کد در زبان‌های برنامه‌نویسی رایج (مثل Python یا JavaScript) در صورت درخواست

۵. توضیح نکات مهم و محدودیت‌های API

همیشه سعی کنید پاسخ‌های خود را با مثال‌های عملی و کاربردی همراه کنید تا درک API برای توسعه‌دهندگان آسان‌تر شود. اگر سوالی نامشخص است، حتماً برای شفاف‌سازی سؤال کنید."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])

        # Create the operation lookup tool
        operation_tool = Tool(
            name="get_operation_spec",
            description="Get detailed OpenAPI specification for a Kenar API operation using its operationId",
            func=self._get_operation_spec,
            return_direct=False
        )
        
        self.tools.append(operation_tool)

        # Initialize the agent with tools
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
            branch="master",
            file_filter=lambda file_path: file_path.endswith(".md")
        )

        # Load the documents - each file will be a single document
        self.github_docs = loader.load()

    def _initialize_vector_stores(self):
        """Initialize two vector stores: one for GitHub docs and one for OpenAPI spec"""

        import json
        from copy import deepcopy

        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_community.vectorstores import FAISS
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings

        # Initialize OpenAI embeddings
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

        # Process GitHub markdown documents
        self.github_vectorstore = FAISS.from_documents(
            documents=self.github_docs,
            embedding=embeddings
        )

        def resolve_ref(ref: str, doc: dict) -> dict:
            """Resolve a JSON reference in the OpenAPI doc"""
            if not ref.startswith('#/'):
                return {}
            
            parts = ref[2:].split('/')
            current = doc
            for part in parts:
                if part in current:
                    current = current[part]
                else:
                    return {}
            return deepcopy(current)

        def format_schema(schema: dict, doc: dict, indent: int = 2) -> str:
            """Format a schema with resolved references and nested objects"""
            if '$ref' in schema:
                schema = resolve_ref(schema['$ref'], doc)
            
            indent_str = " " * indent
            properties = schema.get('properties', {})
            formatted_props = []
            
            for prop_name, prop in properties.items():
                if '$ref' in prop:
                    prop = resolve_ref(prop['$ref'], doc)
                
                prop_type = prop.get('type', 'object')
                base_info = f"{indent_str}{prop_name} ({prop_type})"
                
                if prop.get('description'):
                    base_info += f": {prop['description']}"
                
                if prop_type == 'object':
                    if 'properties' in prop:
                        nested_props = format_schema(prop, doc, indent + 2)
                        if nested_props:
                            formatted_props.append(f"{base_info}:")
                            formatted_props.append(nested_props)
                        else:
                            formatted_props.append(base_info)
                    elif '$ref' in prop:
                        ref_schema = resolve_ref(prop['$ref'], doc)
                        nested_props = format_schema(ref_schema, doc, indent + 2)
                        if nested_props:
                            formatted_props.append(f"{base_info}:")
                            formatted_props.append(nested_props)
                        else:
                            formatted_props.append(base_info)
                    else:
                        formatted_props.append(base_info)
                        
                elif prop_type == 'array' and 'items' in prop:
                    items = prop['items']
                    if '$ref' in items:
                        items = resolve_ref(items['$ref'], doc)
                    
                    if items.get('type') == 'object' and ('properties' in items or '$ref' in items):
                        formatted_props.append(f"{base_info}:")
                        formatted_props.append(f"{indent_str}  Array items:")
                        nested_props = format_schema(items, doc, indent + 4)
                        if nested_props:
                            formatted_props.append(nested_props)
                    else:
                        item_type = items.get('type', 'object')
                        formatted_props.append(f"{base_info} (array of {item_type})")
                else:
                    formatted_props.append(base_info)
            
            return "\n".join(formatted_props)

        # Process OpenAPI documentation with resolved references
        api_texts = []
        
        # Process paths
        paths = self.open_api_doc.get("paths", {})
        for path, methods in paths.items():
            for method, details in methods.items():
                # Resolve parameter references
                parameters = []
                for param in details.get('parameters', []):
                    if '$ref' in param:
                        param = resolve_ref(param['$ref'], self.open_api_doc)
                    parameters.append(param)
                
                param_str = "\n".join([
                    f"  {p.get('name')} ({p.get('in')}): {p.get('description', 'No description')}"
                    for p in parameters
                ])

                # Resolve request body if present
                request_body = ""
                if 'requestBody' in details:
                    body = details['requestBody']
                    if '$ref' in body:
                        body = resolve_ref(body['$ref'], self.open_api_doc)
                    
                    content = body.get('content', {}).get('application/json', {})
                    if 'schema' in content:
                        schema = content['schema']
                        request_body = f"\nRequest Body:\n{format_schema(schema, self.open_api_doc)}"

                # Resolve response references
                responses = details.get('responses', {})
                response_str = []
                for code, resp in responses.items():
                    if '$ref' in resp:
                        resp = resolve_ref(resp['$ref'], self.open_api_doc)
                    
                    content = resp.get('content', {}).get('application/json', {})
                    if 'schema' in content:
                        schema = content['schema']
                        schema_str = format_schema(schema, self.open_api_doc)
                        response_str.append(f"  {code}: {resp.get('description', '')}\n  Response Schema:\n{schema_str}")
                    else:
                        response_str.append(f"  {code}: {resp.get('description', '')}")

                # Build endpoint documentation with only non-empty fields
                doc_parts = [f"Endpoint: {method.upper()} {path}"]
                
                if details.get('summary'):
                    doc_parts.append(f"Summary: {details['summary']}")
                if details.get('description'):
                    doc_parts.append(f"Description: {details['description']}")
                
                # Add parameters section only if there are parameters
                if param_str.strip():
                    doc_parts.append("Parameters:")
                    doc_parts.append(param_str)
                
                # Add request body if present
                if request_body.strip():
                    doc_parts.append(request_body)
                
                # Add responses if present
                if response_str:
                    doc_parts.append("Responses:")
                    doc_parts.append(chr(10).join(response_str))
                
                text = "\n".join(doc_parts)
                api_texts.append(text)

            print("\n\n".join(api_texts))

            # Create Document objects
            from langchain.schema import Document
            api_docs = [
                Document(
                    page_content=text,
                    metadata={"source": "openapi_spec"}
                )
                for text in api_texts
            ]

        # Create vector store directly from the API docs (already split by endpoint)
        self.api_vectorstore = FAISS.from_documents(
            documents=api_docs,
            embedding=embeddings
        )

        # Create combined retriever
        self.retriever = self._create_combined_retriever()

    def _create_combined_retriever(self):
        """Create a combined retriever that searches both vector stores"""
        return lambda query: (
            self.github_vectorstore.similarity_search(query, k=5) +
            self.api_vectorstore.similarity_search(query, k=5)
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
            # Retrieve relevant context using the retriever
            relevant_docs = self.retriever(user_message)
            context = "\n".join([
                f"[{doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
                for doc in relevant_docs
            ])
            
            # Enhance the user message with context
            # Escape any curly braces in the context and user message
            safe_context = context.replace('{', '{{').replace('}', '}}')
            safe_user_message = user_message.replace('{', '{{').replace('}', '}}')
            
            enhanced_message = f"""I have found some potentially relevant information from Kenar documentation that might help answer this question. Feel free to use this context if relevant, but also rely on our conversation history and your general knowledge about APIs and development:

Relevant Documentation:
{safe_context}

User Question: {safe_user_message}

Please provide a helpful answer, incorporating the context where appropriate but maintaining the natural flow of our conversation."""

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
                "input": enhanced_message,
                "chat_history": chat_history
            })

            return response["output"]

        except Exception as e:
            return f"An error occurred: {str(e)}"
