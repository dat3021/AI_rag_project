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
        # if this is with .md in the end 
        if element.type == "blob" and element.path.endswith(".md"):
            try:
                # take raw content
                file_content = repo.get_contents(element.path, ref=branch)
                text_content = file_content.decoded_content.decode("utf-8")
                
                # standardize metadata
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
    pass
    
    if documents:
        print("\n--- first document ---")
        print(f"source: {documents[0].metadata['source']}")
        print(f"content (100 chars): {documents[0].page_content[:100]}...")
