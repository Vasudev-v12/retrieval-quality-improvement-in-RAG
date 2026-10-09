from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate


SYSTEM_PROMPT = """
[INSTRUCTIONS]
1) combine all the text data that is provided to you and answer the questions.
2) do not make up answers from outside the give text data.
"""

class GeminiRAG:

    def __init__(self, api_key):

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-3.6-flash",
            google_api_key=api_key,
            temperature=0
        )

        self.prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
You are a RAG assistant.

Use ONLY the provided context to answer.

Context:
{context}

Question:
{question}

Answer:
"""
        )

    @staticmethod
    def _extract_text(content):
        """Normalize a LangChain message's .content into plain text.

        Some models (e.g. newer Gemini versions) return content as a list of
        content blocks (``[{"type": "text", "text": "...", ...}, ...]``)
        instead of a plain string.
        """
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
            return "".join(parts)
        return str(content)

    def answer(self, question, documents):

        context = "\n\n".join(
            [doc["content"] for doc in documents]
        )

        chain = self.prompt | self.llm

        response = chain.invoke({
            "context": context,
            "question": question
        })

        return self._extract_text(response.content)