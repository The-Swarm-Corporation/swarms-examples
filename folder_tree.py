#!/usr/bin/env python3

from pathlib import Path
from typing import List, Union
from rich.tree import Tree
from rich.console import Console
from rich import print
from rich.panel import Panel
from rich.style import Style


def create_tree(path: Path, tree: Tree) -> None:
    """Recursively build a Tree object representing the directory structure."""
    try:
        # Sort directories first, then files, but only include .py files
        paths = sorted(
            path.iterdir(),
            key=lambda x: (not x.is_dir(), x.name.lower()),
        )

        for item in paths:
            # Skip hidden files and directories
            if item.name.startswith("."):
                continue

            # Only process directories and .py files
            if item.is_dir():
                # Check if directory contains any .py files
                has_py_files = any(
                    f.suffix == ".py" for f in item.rglob("*.py")
                )
                if has_py_files:
                    style = Style(color="red", bold=True)
                    branch = tree.add(f"📁 {item.name}", style=style)
                    create_tree(item, branch)
            elif item.suffix == ".py":
                style = Style(color="bright_red")
                tree.add(f"🐍 {item.name}", style=style)
    except PermissionError:
        tree.add("⚠️ Permission denied", style="red")


def get_file_icon(filename: str) -> str:
    """Return an appropriate emoji icon based on file extension."""
    ext = filename.lower().split(".")[-1] if "." in filename else ""
    icons = {
        "py": "🐍",
        "js": "📜",
        "json": "📋",
        "md": "📝",
        "txt": "📄",
        "yml": "⚙️",
        "yaml": "⚙️",
        "html": "🌐",
        "css": "🎨",
        "png": "🖼️",
        "jpg": "🖼️",
        "jpeg": "🖼️",
        "gif": "🖼️",
        "pdf": "📚",
        "zip": "📦",
        "gz": "📦",
        "tar": "📦",
    }
    return icons.get(ext, "📄")


def display_folder_tree(folders: Union[str, List[str]]) -> None:
    """
    Display a tree structure for one or more folders, showing only Python files.

    Args:
        folders: Either a single folder path as string or a list of folder paths
    """
    console = Console()

    # Convert single string input to list
    if isinstance(folders, str):
        folders = [folders]

    for folder in folders:
        path = Path(folder)
        if not path.exists():
            print(f"[red]❌ Path not found: {folder}[/red]")
            continue

        tree = Tree(
            f"[bold red]📂 {path.absolute()}[/bold red]",
            guide_style="dim black",
        )
        create_tree(path, tree)

        # Create a panel around each tree for better visualization
        panel = Panel(
            tree,
            title=f"[bold red]Python Files: {folder}[/bold red]",
            border_style="red",
            padding=(1, 2),
        )
        console.print(panel)
        console.print()  # Add a blank line between folders


# Example usage
if __name__ == "__main__":
    # Example of using the function with a single folder
    display_folder_tree("examples")

    # Example of using the function with multiple folders
    # display_folder_tree(["examples", "tests", "docs"])
