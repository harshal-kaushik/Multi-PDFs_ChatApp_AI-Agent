from transformers import pipeline

from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough,RunnableParallel,RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from config import (
    LLM_MODEL,
    MAX_LENGTH,
    TEMPERATURE,
    TOP_K
)

from vector_store import load_vector_store


def load_llm():

    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=TEMPERATURE,
        max_output_tokens=MAX_LENGTH
    )

    return llm


def format_docs(docs):
    return "\n\n".join(
        doc.page_content for doc in docs
    )


def get_prompt():
    template = """
    Answer the question only from the provided context.

    If the answer is not available in the context,
    say:
    "answer is not available in the context"

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

    return prompt


def get_rag_chain():
    db = load_vector_store()

    retriever = db.as_retriever(search_type="similarity",
        search_kwargs={"k": TOP_K}
    )

    prompt = get_prompt()
    llm = load_llm()

    parallel_chain = RunnableParallel({
        'context': retriever | RunnableLambda(format_docs),
        'question': RunnablePassthrough()
    })

    final_chain = (
            parallel_chain
            | prompt
            | llm
            | StrOutputParser()
    )

    return final_chain


def ask_question(user_question):
    chain = get_rag_chain()

    response = chain.invoke(
        user_question
    )

    return response