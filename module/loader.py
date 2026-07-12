import os
from github import Github
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()
def load_markdown_from_github(repo_owner: str, repo_name: str, branch: str = "main"):
    github_token = os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN")
    
    print(f"access {repo_owner}/{repo_name}...")
    
    g = Github(github_token) if github_token else Github()
    
    try:
        repo = g.get_repo(f"{repo_owner}/{repo_name}")
        tree = repo.get_git_tree(branch, recursive=True)
    except Exception as e:
        print(f"error: {e}")
        return []

    docs = []
    
    for element in tree.tree:
        # Nếu là file (blob) và đuôi .md
        if element.type == "blob" and element.path.endswith(".md"):
            try:
                # Lấy nội dung file thô
                file_content = repo.get_contents(element.path, ref=branch)
                text_content = file_content.decoded_content.decode("utf-8")
                
                # Biến đổi thành chuẩn Document của Langchain để truyền đi bước sau
                doc = Document(
                    page_content=text_content,
                    metadata={
                        "source": element.path,
                        "repo": f"{repo_owner}/{repo_name}",
                        "branch": branch
                    }
                )
                docs.append(doc)
            except Exception as e:
                print(f"error: {e}")

    print(f"Total: {len(docs)} LangChain Documents.")
    return docs

if __name__ == "__main__":
    OWNER = "dat3021" 
    REPO = "RAG_document"
    
    documents = load_markdown_from_github(repo_owner=OWNER, repo_name=REPO)
    
    if documents:
        print("\n--- FILE ĐẦU TIÊN TÌM ĐƯỢC ---")
        print(f"Nguồn: {documents[0].metadata['source']}")
        print(f"Nội dung (100 ký tự): {documents[0].page_content[:100]}...")