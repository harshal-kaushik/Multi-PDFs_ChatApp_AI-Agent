import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
import re
from io import BytesIO
from PyPDF2 import PdfReader
from pptx import Presentation
from pptx.util import Inches, Pt
import uuid

from config import (
    LLM_MODEL,
    MAX_LENGTH,
    TEMPERATURE,
    TOP_K,
    CHUNK_SIZE_CHILD,CHUNK_OVERLAP_CHILD,
    CHUNK_OVERLAP_PARENT,CHUNK_SIZE_PARENT,
    EMBEDDING_MODEL,
    FAISS_PATH
)

class RagEngine:
    def __init__(self, db_store: str = FAISS_PATH):
        # Initializing core models using parameters loaded from config.py
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={
                'normalize_embeddings': True
            }
        )
        self.llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            temperature=TEMPERATURE,
            max_output_tokens=MAX_LENGTH
        )
        ## changes of day-3 changing chunking to parent-child chuks -->> crating two seperate splitter one for the parent chunk and one for the child chunk
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE_PARENT,
            chunk_overlap=CHUNK_OVERLAP_PARENT,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                ""
            ]
        )
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE_CHILD,
            chunk_overlap=CHUNK_OVERLAP_CHILD,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                ""
            ]
        )
        self.db_path = db_store
        self.vector_store = self._load_existing_store()

    def _load_existing_store(self):
        """Loads the FAISS index from disk if it exists, preserving state across restarts."""
        if os.path.exists(self.db_path):
            try:
                print(f"Loading existing FAISS index from {self.db_path}...")
                return FAISS.load_local(
                    self.db_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True  # Required by LangChain to safely deserialize pickling fields
                )
            except Exception as e:
                print(f"Failed to load existing index: {e}. Starting fresh.")
                return None
        return None

    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extracts plain text out of a target PDF using PyPDF2."""
        text = ""
        pdf_reader = PdfReader(file_path)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        return text

    def ingest_pdf(self, file_path: str):
        """Extracts plain text strings, structures Parent-Child metadata links, and indexes documents."""
        raw_text = self.extract_text_from_pdf(file_path)

        if not raw_text.strip():
            raise ValueError("The uploaded PDF seems to be empty or unscannable.")

        # split documnet into parent text block
        parent_texts = self.parent_splitter.split_text(raw_text)
        child_documents = []

        # processing each parent text blog
        for parent_text in parent_texts:
            # Generate a completely unique identifier string for this parent block
            parent_id = str(uuid.uuid4())

            # Slice that SPECIFIC parent block into tiny child fragments
            sub_chunks = self.child_splitter.split_text(parent_text)

            # 3. we are wrapping each child segment into langchain document with there parent id and parent metadata
            for chunk in sub_chunks:
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "parent_id": parent_id,
                        "parent_content": parent_text  # Inlined to keep local FAISS persistence seamless
                    }
                )
                child_documents.append(doc)
        # Adding the child documnet directly  to the vector store
        if self.vector_store is None:
            self.vector_store = FAISS.from_texts(child_documents, self.embeddings)
        else:
            self.vector_store.add_texts(child_documents)

        self.vector_store.save_local(self.db_path)
        return len(child_documents)

    def _format_docs(self, docs):
        """Formats retrieved chunks into a single text block for the LLM prompt context."""
        return "\n\n".join(doc.page_content for doc in docs)

    def _resolve_parent_context(self,retrived_child_documents):
        """Intercept the matched child chunks , descrite if there any duplicste present and get the full parent context"""

        seen_parent_ids = set()
        parent_blocks = []


        for doc in retrived_child_documents:
            parent_id = doc.metadata.get("parent_id")
            parent_content = doc.metadata.get("parent_content")

            if parent_id not in seen_parent_ids and parent_id :
                seen_parent_ids.add(parent_id)
                parent_blocks.append(parent_content)
        # joining all parent blocks with a clear seperate boundary line
            return "\n\n--- NEXT SECTION BOUNDARY ---\n\n".join(parent_blocks)


    def _get_slide_prompt(self):
        template = """
    You are an expert executive business analyst. Your job is to extract high-value insights from the context below and structure them into a professional, cohesive 5 to 7 slide presentation deck based on the request: "{question}".

    Guidelines:
    - Slide 1 must always be an executive Title Slide.
    - Every slide must focus on a distinct business pillar, timeline step, metric, or strategic insight.
    - Provide exactly 3 to 5 highly concise, impactful bullet points per content slide.

    Strict Output Format Requirement:
    You must structure your entire output using the exact layout schema block shown below. Do not use conversational introductions, markdown bolding highlights, or trailing notes. Begin directly with [SLIDE].

    [SLIDE]
    TITLE: Executive Summary & Overview
    BULLET: Extracted business metric showing market position
    BULLET: Key strategic priority identified from document resources
    BULLET: Actionable takeaway for business leaders

    [SLIDE]
    TITLE: Next Slide Heading
    BULLET: Next slide point

    Context:
    {context}

    Question/Topic Request:
    {question}

    Structured Presentation Content:
    """
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

    def generation_presentation_text(self, user_question: str) -> str:
        """Assembles your exact parallel pipeline workflow dynamically and triggers execution."""
        if self.vector_store is None:
            return "No documents have been indexed yet. Please upload a PDF file first."

        # Initialize the retriever dynamically using your TOP_K configuration parameter
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": TOP_K}
        )

        # Build your exact parallel LCEL chain mapping
        parallel_chain = RunnableParallel({
            'context': retriever | RunnableLambda(self._resolve_parent_context),
            'question': RunnablePassthrough()
        })

        slide_chain = (
            parallel_chain
            | self._get_slide_prompt()
            | self.llm
            | StrOutputParser()
        )

        # Trigger execution and return the string answer
        return slide_chain.invoke(user_question)

    # ANOTHER FUNCTION CREATED TO CONVERT THE TEXT OUTPUT INTO DOCUMENTED SLIDES
    def export_to_pptx(self, raw_llm_text: str) -> BytesIO:
        """Parses raw text separated by [SLIDE] tags and maps them to clean widescreen slides."""
        prs = Presentation()
        prs.slide_width = Inches(13.33)  # 16:9 Aspect Ratio
        prs.slide_height = Inches(7.5)

        # Break up the response text block by block
        slide_raw_blocks = raw_llm_text.split("[SLIDE]")

        for block in slide_raw_blocks:
            if not block.strip():
                continue

            lines = [line.strip() for line in block.split("\n") if line.strip()]

            slide_title = "Executive Summary"
            bullet_points = []

            # Line by line structural mapping
            for line in lines:
                if line.upper().startswith("TITLE:"):
                    slide_title = line[6:].strip()
                elif line.upper().startswith("BULLET:"):
                    bullet_points.append(line[7:].strip())

            # Using standard blank layout configuration
            blank_layout = prs.slide_layouts[6]
            slide = prs.slides.add_slide(blank_layout)

            # 1. Build Title Box (Positioned elegantly at top)
            title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.6), Inches(11.7), Inches(1.2))
            tf_title = title_box.text_frame
            tf_title.word_wrap = True
            p_title = tf_title.paragraphs[0]
            p_title.text = slide_title
            p_title.font.size = Pt(38)
            p_title.font.bold = True
            p_title.font.name = "Arial"

            # 2. Build Content Box (Positioned cleanly for body text)
            content_box = slide.shapes.add_textbox(Inches(0.8), Inches(2.2), Inches(11.7), Inches(4.5))
            tf_content = content_box.text_frame
            tf_content.word_wrap = True

            for i, bp_text in enumerate(bullet_points):
                # Recycle the default first paragraph, add new ones after
                p = tf_content.paragraphs[0] if i == 0 else tf_content.add_paragraph()
                p.text = bp_text
                p.font.size = Pt(18)
                p.font.name = "Arial"
                p.level = 0  # Automatically triggers native PPT bullet indentation styles
                p.space_after = Pt(14)

        # Compress and save structural slides to a memory byte buffer stream
        output_stream = BytesIO()
        prs.save(output_stream)
        output_stream.seek(0)
        return output_stream
