from pathlib import Path
import os

def get_project_root():
    """Find project root in both local and Colab environments."""
    current = Path.cwd()
    
    # Check if we're already in PROJECT
    if current.name == "PROJECT":
        return current
    
    # Look up through parents
    for parent in current.parents:
        if parent.name == "PROJECT":
            return parent
    
    # Fallback for Colab: check if we're in mounted Drive
    if "/content/drive" in str(current):
        project_path = Path('/content/drive/MyDrive/PROJECT')
        if project_path.exists():
            return project_path
    
    raise FileNotFoundError("Project root 'PROJECT' not found in path")


def display_path(path: str | Path) -> Path:
    path_str = str(path)
    if "sqlite:/" in path_str:
        path_str = path_str.split("sqlite:/")[-1]
        if not path_str.startswith("/"):
            path_str = "/" + path_str
    return Path(path_str).relative_to(get_project_root().parent)