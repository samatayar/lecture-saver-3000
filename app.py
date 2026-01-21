import streamlit as st
import os
import tempfile
import shutil
from pathlib import Path
import json
import re
from typing import List, Dict, Any
import time

# Import RAG pipeline components
from docling.document_converter import DocumentConverter
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from openai import OpenAI
import numpy as np

# Configure page
st.set_page_config(
    page_title="📚 Document Chat Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'collection' not in st.session_state:
    st.session_state.collection = None

if 'chroma_client' not in st.session_state:
    st.session_state.chroma_client = None

# API Configuration - will be accessed when needed
def get_api_keys():
    """Get API keys from secrets"""
    try:
        openrouter_key = st.secrets.get("OPENROUTER_API_KEY", "")
        huggingface_key = st.secrets.get("HUGGINGFACE_TOKEN", "")

        # Set environment variables as backup
        if openrouter_key:
            os.environ["OPENROUTER_API_KEY"] = openrouter_key
        if huggingface_key:
            os.environ["HUGGINGFACE_TOKEN"] = huggingface_key

    except Exception as e:
        print(f"Warning: Could not access st.secrets: {e}")
        # Fallback to environment variables
        openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")
        huggingface_key = os.environ.get("HUGGINGFACE_TOKEN", "")

    # If no key found, use the provided OpenRouter key
    if not openrouter_key:
        openrouter_key = "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"

    return {
        "openrouter": openrouter_key,
        "huggingface": huggingface_key
    }

# Document class for chunking
class Document:
    def __init__(self, page_content: str, metadata: dict = None):
        self.page_content = page_content
        self.metadata = metadata or {}

def extract_pdf_to_markdown(pdf_path: str) -> str | None:
    """Extract text from PDF and convert to markdown"""
    pdf_file = Path(pdf_path)

    if not pdf_file.exists() or not pdf_file.is_file():
        st.error(f"File not found or not a file: {pdf_path}")
        return None

    with st.spinner(f"Processing: {pdf_file.name}"):
        try:
            converter = DocumentConverter()
            result = converter.convert(str(pdf_file))
            doc = result.document

            raw_md = doc.export_to_markdown(
                page_break_placeholder="\n\n---\n\n"
            )

            cleaned = re.sub(r'<!--.*?-->', '', raw_md, flags=re.DOTALL)
            cleaned = re.sub(r'</?image[^>]*>', '', cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
            cleaned = re.sub(r'\s*---\s*', '\n\n---\n\n', cleaned)
            cleaned = cleaned.strip()

            # Create temp directory for markdown files
            temp_dir = Path("temp_md")
            temp_dir.mkdir(exist_ok=True)

            output_file = temp_dir / f"{pdf_file.stem}_markdown.md"
            output_file.write_text(cleaned, encoding='utf-8')

            st.success(f"Successfully processed: {pdf_file.name}")
            return str(output_file)

        except Exception as e:
            st.error(f"Error processing {pdf_file.name}: {str(e)}")
            return None

def describe_tables_and_figures(text: str) -> str:
    """Convert tables and figures to descriptive text"""
    md_table_pattern = re.compile(r"(\|.+\|\n\|[-:\s|]+\|\n(?:\|.*\|\n?)*)", re.MULTILINE)
    def md_table_to_text(match):
        table = match.group(1)
        rows = [r.strip() for r in table.splitlines() if r.strip()]
        header = rows[0] if rows else ""
        data_rows = rows[2:] if len(rows) > 2 else []
        return f"\n[جدول]: يحتوي على بيانات منظمة. العناوين: {header}. عدد الصفوف: {len(data_rows)}.\n"
    text = md_table_pattern.sub(md_table_to_text, text)
    text = re.sub(r"<table.*?>.*?</table>", "\n[جدول]: جدول HTML يحتوي على بيانات منظمة.\n", text, flags=re.DOTALL|re.IGNORECASE)
    text = re.sub(r"```mermaid.*?```", "\n[مخطط]: مخطط Mermaid يوضح علاقات أو تدفق عمليات.\n", text, flags=re.DOTALL)
    text = re.sub(r"!\[.*?\]\(.*?\)", "\n[شكل]: شكل أو مخطط بصري.\n", text)
    return text

def merge_empty_sections(docs: List[Document]) -> List[Document]:
    """Merge empty sections in documents"""
    merged_docs = []
    buffer_doc = None
    for doc in docs:
        content = doc.page_content.strip()
        if not content:
            if buffer_doc is not None:
                buffer_doc.page_content += "\n" + content
            else:
                buffer_doc = doc
            continue
        if buffer_doc is not None:
            buffer_doc.page_content += "\n" + content
            merged_docs.append(buffer_doc)
            buffer_doc = None
        else:
            merged_docs.append(doc)
    if buffer_doc is not None:
        merged_docs.append(buffer_doc)
    return merged_docs

def split_md_files_to_json(
    md_files: List[str],
    output_dir: str,
    chunk_size: int = 900,
    chunk_overlap: int = 120
):
    """Split markdown files into chunks and save as JSON"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")])
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    all_chunks = []

    for md_file in md_files:
        md_path = Path(md_file)
        raw_text = md_path.read_text(encoding="utf-8")
        pages = raw_text.split("\n---\n")

        for page_num, page_text in enumerate(pages, start=1):
            clean_text = describe_tables_and_figures(page_text)
            header_docs = header_splitter.split_text(clean_text)
            header_docs = merge_empty_sections(header_docs)

            for doc in header_docs:
                header = doc.metadata.get("h3") or doc.metadata.get("h2") or doc.metadata.get("h1")
                chunks = text_splitter.split_text(doc.page_content)
                for chunk in chunks:
                    all_chunks.append({
                        "text": chunk.strip(),
                        "metadata": {
                            "source": md_path.name,
                            "page": page_num,
                            "header": header,
                        }
                    })

    out_file = output_path / f"chunks.json"
    out_file.write_text(json.dumps(all_chunks, ensure_ascii=False, indent=2), encoding="utf-8")

    return str(out_file)

def clean_text(text: str) -> str:
    """Clean text for embedding"""
    text = re.sub(r'```.*?```', ' ', text, flags=re.DOTALL)
    text = re.sub(r'`([^`]+)`', r'\1', text)
    text = re.sub(r'#{1,6}\s+', '', text)
    text = re.sub(r'^\s*>\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*[\*\-\+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\*\*([^\*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^\*]+)\*', r'\1', text)
    text = re.sub(r'__([^_]+)__', r'\1', text)
    text = re.sub(r'_([^_]+)_', r'\1', text)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    text = re.sub(r'https?://[^\s]+', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\|[\s\-\:]+\|', ' ', text)
    text = re.sub(r'\|', ' ', text)
    text = re.sub(r'([!?.]){3,}', r'\1', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = '\n'.join(line.strip() for line in text.split('\n'))
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def generate_embeddings(input_path: str | Path, output_dir: str | Path, model_name: str = "sentence-transformers/distiluse-base-multilingual-cased-v2"):
    """Generate embeddings for text chunks"""
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize Hugging Face if token is available
    api_keys = get_api_keys()
    if api_keys["huggingface"]:
        from huggingface_hub import login
        login(api_keys["huggingface"])

    with st.spinner("Loading embedding model..."):
        model = SentenceTransformer(model_name)

    json_files = [input_path] if input_path.is_file() else list(input_path.glob("*.json"))

    all_results = []
    for json_file in json_files:
        with open(json_file, encoding="utf-8") as f:
            chunks = json.load(f)

        for i, chunk in enumerate(chunks, start=1):
            text = chunk.get("text") or chunk.get("text_content", "")
            text = clean_text(text)
            if not text:
                continue

            vector = model.encode(text, convert_to_numpy=True)
            vector = vector.astype(float).tolist()

            all_results.append({
                "id": f"chunk_{len(all_results) + 1}",
                "vector": vector,
                "text_content": text,
                "metadata": chunk.get("metadata", {})
            })

    out_file = output_dir / "embeddings.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    return all_results

def initialize_chroma_db(db_path: str = "./chromaDB", collection_name: str = "Final"):
    """Initialize ChromaDB client and collection"""
    if st.session_state.chroma_client is None:
        st.session_state.chroma_client = chromadb.PersistentClient(path=db_path)

    if st.session_state.collection is None:
        try:
            st.session_state.collection = st.session_state.chroma_client.get_collection(name=collection_name)
        except:
            st.session_state.collection = st.session_state.chroma_client.create_collection(name=collection_name)

    return st.session_state.collection

def check_existing_collections(db_path: str = "./chromaDB"):
    """Check for existing collections in ChromaDB"""
    try:
        client = chromadb.PersistentClient(path=db_path)
        collections = client.list_collections()
        return [col.name for col in collections]
    except Exception as e:
        st.warning(f"خطأ في التحقق من قاعدة البيانات: {str(e)}")
        return []

def get_collection_stats(collection):
    """Get statistics about a collection"""
    try:
        count = collection.count()
        return count
    except:
        return 0

def load_embeddings_to_chroma(embeddings: List[Dict], collection_name: str = "Final"):
    """Load embeddings into ChromaDB"""
    collection = initialize_chroma_db(collection_name=collection_name)

    def clean_metadata(metadata: dict) -> dict:
        return {k: ("" if v is None else str(v)) for k, v in metadata.items()}

    for chunk in embeddings:
        try:
            collection.add(
                ids=[chunk["id"]],
                documents=[chunk["text_content"]],
                embeddings=[chunk["vector"]],
                metadatas=[clean_metadata(chunk["metadata"])]
            )
        except Exception as e:
            st.warning(f"Skipping duplicate chunk {chunk['id']}: {str(e)}")

    return collection

def import_json_to_collection(json_file_path: str, collection_name: str):
    """Import embeddings from JSON file to a specific collection"""
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            embeddings = json.load(f)

        if not embeddings:
            return False, "الملف فارغ أو لا يحتوي على بيانات صحيحة"

        collection = load_embeddings_to_chroma(embeddings, collection_name)
        return True, f"تم استيراد {len(embeddings)} قطعة بنجاح"

    except FileNotFoundError:
        return False, "الملف غير موجود"
    except json.JSONDecodeError:
        return False, "خطأ في قراءة ملف JSON"
    except Exception as e:
        return False, f"خطأ غير متوقع: {str(e)}"

def query_chroma(
    query: str,
    collection,
    model_name: str = "sentence-transformers/distiluse-base-multilingual-cased-v2",
    n_results: int = 10
):
    """Query ChromaDB for relevant documents"""
    model = SentenceTransformer(model_name)
    query_vector = model.encode([query], convert_to_numpy=True)[0].astype(float).tolist()

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=n_results
    )

    output = []
    for i, doc in enumerate(results["documents"][0]):
        output.append({
            "text": doc,
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i]
        })

    return output

def format_context(chunks):
    """Format retrieved chunks for LLM context"""
    return "\n\n".join(
        f"[Source: {c['metadata'].get('source')} | Page: {c['metadata'].get('page')}]\n{c['text']}"
        for c in chunks
    )

def rag_answer(query, retrieved_chunks):
    """Generate answer using RAG with OpenRouter"""
    api_keys = get_api_keys()

    if not api_keys["openrouter"]:
        return "❌ API key not configured. Please set OPENROUTER_API_KEY in secrets."

    context = format_context(retrieved_chunks)

    try:
        client = OpenAI(
            api_key=api_keys["openrouter"],
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "https://streamlit-app.com/",
                "X-Title": "Document Chat Assistant"
            }
        )

        response = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "أنت مساعد ذكي متخصص في الإجابة عن الأسئلة بناءً على الوثائق المقدمة.\n"
                        "القواعد المهمة:\n"
                        "• أجب دائمًا بنفس اللغة التي كُتب بها السؤال.\n"
                        "  - إذا كان السؤال بالعربية → أجب بالعربية الفصحى الواضحة\n"
                        "  - إذا كان السؤال بالإنجليزية → أجب بالإنجليزية الواضحة\n"
                        "• استخدم المعلومات الموجودة في الـ Context فقط.\n"
                        "• لا تُضف معلومات من خارج السياق.\n"
                        "• إذا لم يكن الجواب موجودًا في السياق، قل: «لا أعرف» أو «لا توجد معلومات كافية في السياق».\n"
                        "• كن موجزًا ودقيقًا ومباشرًا."
                    )
                },
                {
                    "role": "user",
                    "content": f"""
Context:
{context}

Question:
{query}
"""
                }
            ],
            temperature=0.2
        )

        answer = response.choices[0].message.content

        # Extract unique sources
        sources = []
        seen_sources = set()
        for i, chunk in enumerate(retrieved_chunks, start=1):
            source = chunk['metadata'].get('source', 'Unknown')
            page = chunk['metadata'].get('page', 'N/A')
            source_key = f"{source} (Page {page})"
            if source_key not in seen_sources:
                seen_sources.add(source_key)
                sources.append(f"{i}. {source_key}")

        formatted_output = f"""
💡 **الإجابة:**
{answer}

🔍 **المصادر:**
{chr(10).join(sources) if sources else 'لا توجد مصادر متاحة.'}
"""

        return formatted_output

    except Exception as e:
        return f"❌ خطأ في إنشاء الإجابة: {str(e)}"

def process_uploaded_file(uploaded_file):
    """Process an uploaded file through the complete RAG pipeline"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        # Step 1: Extract PDF to markdown
        md_file = extract_pdf_to_markdown(tmp_path)
        if not md_file:
            return False

        # Step 2: Split into chunks
        temp_chunks_dir = "temp_chunks"
        chunks_file = split_md_files_to_json([md_file], temp_chunks_dir)

        # Step 3: Generate embeddings
        temp_embeddings_dir = "temp_embeddings"
        embeddings = generate_embeddings(chunks_file, temp_embeddings_dir)

        # Step 4: Load to ChromaDB
        collection = load_embeddings_to_chroma(embeddings)

        # Track processed file
        st.session_state.processed_files.append({
            "name": uploaded_file.name,
            "chunks_count": len(embeddings),
            "processed_at": time.time()
        })

        st.success(f"✅ تمت معالجة الملف '{uploaded_file.name}' بنجاح! ({len(embeddings)} قطعة)")

        # Cleanup temp files
        os.unlink(tmp_path)
        shutil.rmtree("temp_md", ignore_errors=True)
        shutil.rmtree(temp_chunks_dir, ignore_errors=True)
        shutil.rmtree(temp_embeddings_dir, ignore_errors=True)

        return True

    except Exception as e:
        st.error(f"❌ خطأ في معالجة الملف: {str(e)}")
        os.unlink(tmp_path)
        return False

def main():
    st.title("📚 مساعد الدردشة مع الوثائق")
    st.markdown("---")

    # Check for existing data on app startup
    if 'available_collections' not in st.session_state:
        st.session_state.available_collections = check_existing_collections()

    if 'selected_collection' not in st.session_state:
        if st.session_state.available_collections:
            # Prefer "Final" collection if it exists, otherwise use the first available
            if "Final" in st.session_state.available_collections:
                st.session_state.selected_collection = "Final"
            else:
                st.session_state.selected_collection = st.session_state.available_collections[0]
        else:
            st.session_state.selected_collection = "Final"

    # Sidebar for file uploads and settings
    with st.sidebar:
        st.header("📤 رفع الملفات")

        uploaded_files = st.file_uploader(
            "اختر ملفات PDF للتحليل",
            type=["pdf"],
            accept_multiple_files=True,
            help="يمكنك رفع ملفات PDF متعددة"
        )

        if uploaded_files:
            if st.button("🔄 معالجة الملفات", type="primary"):
                with st.spinner("جاري معالجة الملفات..."):
                    initialize_chroma_db()

                    success_count = 0
                    for uploaded_file in uploaded_files:
                        if process_uploaded_file(uploaded_file):
                            success_count += 1

                    if success_count > 0:
                        st.success(f"تمت معالجة {success_count} من {len(uploaded_files)} ملف بنجاح!")
                        # Refresh available collections
                        st.session_state.available_collections = check_existing_collections()
                        st.rerun()

        # Database Collections Management
        st.header("🗄️ قواعد البيانات")

        # Create new collection
        with st.expander("➕ إنشاء مجموعة جديدة"):
            new_collection_name = st.text_input(
                "اسم المجموعة الجديدة:",
                placeholder="مثال: lectures_arabic",
                help="سيتم إنشاء مجموعة منفصلة للبيانات الجديدة"
            )
            if st.button("إنشاء مجموعة جديدة", type="secondary"):
                if new_collection_name and new_collection_name not in st.session_state.available_collections:
                    try:
                        client = chromadb.PersistentClient(path="./chromaDB")
                        client.create_collection(name=new_collection_name)
                        st.session_state.available_collections = check_existing_collections()
                        st.session_state.selected_collection = new_collection_name
                        st.session_state.collection = None
                        st.success(f"✅ تم إنشاء المجموعة: {new_collection_name}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ خطأ في إنشاء المجموعة: {str(e)}")
                elif new_collection_name in st.session_state.available_collections:
                    st.warning("المجموعة موجودة بالفعل!")

        # Import from JSON
        with st.expander("📥 استيراد من ملف JSON"):
            json_file = st.text_input(
                "مسار ملف JSON:",
                placeholder="مثال: ./embeddings.json",
                help="مسار ملف JSON يحتوي على embeddings"
            )
            import_collection_name = st.text_input(
                "اسم المجموعة المستهدفة:",
                value=st.session_state.selected_collection,
                help="سيتم استيراد البيانات إلى هذه المجموعة"
            )
            if st.button("استيراد البيانات", type="secondary"):
                if json_file and import_collection_name:
                    with st.spinner("جاري استيراد البيانات..."):
                        success, message = import_json_to_collection(json_file, import_collection_name)
                        if success:
                            st.success(f"✅ {message}")
                            st.session_state.available_collections = check_existing_collections()
                            st.rerun()
                        else:
                            st.error(f"❌ {message}")
                else:
                    st.warning("يرجى تحديد مسار الملف واسم المجموعة")

        if st.session_state.available_collections:
            st.subheader("📚 المجموعات المتاحة")
            selected = st.selectbox(
                "اختر مجموعة البيانات:",
                options=st.session_state.available_collections,
                index=st.session_state.available_collections.index(st.session_state.selected_collection) if st.session_state.selected_collection in st.session_state.available_collections else 0,
                key="collection_selector"
            )

            if selected != st.session_state.selected_collection:
                st.session_state.selected_collection = selected
                st.session_state.collection = None  # Reset collection to load new one
                st.rerun()

            # Show collection stats and management
            try:
                collection = initialize_chroma_db(collection_name=st.session_state.selected_collection)
                doc_count = get_collection_stats(collection)
                st.info(f"📊 تحتوي على {doc_count} قطعة من البيانات")

                # Collection management options
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("🗑️ حذف المجموعة", type="secondary", help="سيتم حذف جميع البيانات في هذه المجموعة"):
                        try:
                            client = chromadb.PersistentClient(path="./chromaDB")
                            client.delete_collection(st.session_state.selected_collection)
                            st.session_state.available_collections = check_existing_collections()
                            if st.session_state.available_collections:
                                st.session_state.selected_collection = st.session_state.available_collections[0]
                            else:
                                st.session_state.selected_collection = "Final"
                            st.session_state.collection = None
                            st.success("✅ تم حذف المجموعة بنجاح")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ خطأ في حذف المجموعة: {str(e)}")

                with col2:
                    if st.button("📋 عرض تفاصيل", type="secondary"):
                        try:
                            results = collection.peek(limit=5)
                            st.write("**عينة من البيانات:**")
                            for i, doc in enumerate(results['documents']):
                                st.write(f"**الوثيقة {i+1}:**")
                                st.write(doc[:200] + "..." if len(doc) > 200 else doc)
                                st.write("---")
                        except Exception as e:
                            st.error(f"خطأ في عرض التفاصيل: {str(e)}")

                with col3:
                    if st.button("🔄 تحديث", type="secondary", help="تحديث قائمة المجموعات"):
                        st.session_state.available_collections = check_existing_collections()
                        st.success("✅ تم تحديث قائمة المجموعات")
                        st.rerun()

            except Exception as e:
                st.error(f"خطأ في تحميل المجموعة: {str(e)}")
        else:
            st.info("📭 لا توجد مجموعات بيانات محفوظة")
            st.session_state.selected_collection = "documents"

        # Show processed files
        if st.session_state.processed_files:
            st.header("📋 الملفات المعالجة")
            for file_info in st.session_state.processed_files:
                with st.expander(f"📄 {file_info['name']}"):
                    st.write(f"**عدد القطع:** {file_info['chunks_count']}")
                    st.write(f"**تاريخ المعالجة:** {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(file_info['processed_at']))}")

        # Settings
        st.header("⚙️ الإعدادات")
        n_results = st.slider("عدد النتائج المسترجعة", min_value=5, max_value=20, value=10)

    # Main chat interface
    st.header("💬 الدردشة مع الوثائق")

    # Check if we have any data to work with
    has_processed_files = bool(st.session_state.processed_files)
    has_existing_collections = bool(st.session_state.available_collections)

    if not has_processed_files and not has_existing_collections:
        st.info("📤 يرجى رفع ومعالجة ملفات PDF أولاً من الشريط الجانبي")
        return

    if has_existing_collections:
        collection = initialize_chroma_db(collection_name=st.session_state.selected_collection)
        doc_count = get_collection_stats(collection)
        st.success(f"✅ متصل بقاعدة البيانات: **{st.session_state.selected_collection}** ({doc_count} قطعة)")
    elif has_processed_files:
        st.info("💡 يمكنك الدردشة مع الملفات المرفوعة مؤخراً")

    # Initialize ChromaDB if needed
    collection = initialize_chroma_db(collection_name=st.session_state.selected_collection)

    # Chat input
    query = st.text_input(
        "اكتب سؤالك هنا:",
        placeholder="مثال: ما هي أهداف أمن المعلومات؟",
        help="يمكنك السؤال بالعربية أو الإنجليزية"
    )

    if query and st.button("🚀 اسأل", type="primary"):
        with st.spinner("جاري البحث والإجابة..."):
            # Retrieve relevant chunks
            results = query_chroma(query, collection, n_results=n_results)

            if not results:
                st.warning("لم يتم العثور على نتائج ذات صلة")
                return

            # Generate answer
            answer = rag_answer(query, results)

            # Add to chat history
            st.session_state.chat_history.append({
                "query": query,
                "answer": answer,
                "timestamp": time.time(),
                "sources_count": len(results)
            })

    # Display chat history
    if st.session_state.chat_history:
        st.markdown("---")
        st.subheader("📝 سجل المحادثات")

        for i, chat in enumerate(reversed(st.session_state.chat_history[-10:])):  # Show last 10
            with st.expander(f"❓ {chat['query']} ({time.strftime('%H:%M:%S', time.localtime(chat['timestamp']))})", expanded=(i==0)):
                st.markdown(chat['answer'])

    # Footer
    st.markdown("---")
    st.markdown("*بني باستخدام Streamlit و ChromaDB و OpenRouter*")

if __name__ == "__main__":
    main()