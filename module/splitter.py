from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def split_markdown_documents(documents: list[Document], chunk_size: int = 1000, chunk_overlap: int = 100) -> list[Document]:
    """
    Splits a list of LangChain markdown Documents into smaller chunks.
    It first splits semantically based on markdown headers, then applies
    a character-based splitter for chunks that are still too large.
    """
    
    # 1. First Pass: Split by Markdown Headers
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on, strip_headers=False)
    
    md_header_splits = []
    for doc in documents:
        # Split the text content of the document
        splits = markdown_splitter.split_text(doc.page_content)
        
        # We need to manually add the original metadata (like source, repo) back to each new chunk
        for split in splits:
            split.metadata.update(doc.metadata)
            md_header_splits.append(split)

    # 2. Second Pass: Split any remaining large chunks by character count
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    
    final_splits = text_splitter.split_documents(md_header_splits)
    
    print(f"Original docs: {len(documents)}. After chunking: {len(final_splits)} chunks.")
    return final_splits


if __name__ == "__main__":
    # Quick test logic
    doc = Document(
        page_content="# Foo\n\n## Bar\n\nHi this is Jim\n\nHi this is Joe\n\n### Boo \n\nHi this is Lance \n\n## Baz\n\nHi this is Molly",
        metadata={"source": "test.md"}
    )
    
    splits = split_markdown_documents([doc], chunk_size=50, chunk_overlap=10)
    for i, s in enumerate(splits):
        print(f"--- Chunk {i} ---")
        print(s.metadata)
        print(s.page_content)