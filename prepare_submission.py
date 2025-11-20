#!/usr/bin/env python3
"""
Script to help prepare project submission by listing files to include/exclude.
Run: python prepare_submission.py
"""

import os
from pathlib import Path

# Directories to exclude (both exact names and full paths)
EXCLUDE_DIRS = {
    '__pycache__',
    'user_videos_cache',
    'references',
    'output',  # demo/output
    'video',   # demo/video
    'temp_user_processing',
    'temp_processing',
    'checkpoint',
    '.git',
    '.vscode',
    '.idea',
    'venv',
    'env',
    '.conda',
}

# Additional path patterns to exclude
EXCLUDE_PATH_PATTERNS = [
    'demo/output',
    'demo/video',
    'demo/lib/checkpoint',
]

# File patterns to exclude
EXCLUDE_PATTERNS = {
    '*.pyc',
    '*.pyo',
    '*.pth',
    '*.pt',
    '*.bin',
    '*.weights',
    '*.mp4',
    '*.avi',
    '*.mov',
    '*.gif',
    '*.npz',
    '*.log',
    '*.json',
    'comparison_*.mp4',
    'comparison_*.gif',
}

# Files to always include (even if they match exclude patterns)
ALWAYS_INCLUDE = {
    'requirements.txt',
    '.gitignore',
    'README.md',
    'README_FITNESS.md',
    'GITHUB_SETUP.md',
    'VIDEO_COMPARISON_GUIDE.md',
    'LICENSE',
    'SUBMISSION_GUIDE.md',
}

def should_exclude_file(file_path: Path) -> bool:
    """Check if a file should be excluded."""
    file_name = file_path.name
    
    # Always include certain files
    if file_name in ALWAYS_INCLUDE:
        return False
    
    # Check if file matches exclude patterns
    for pattern in EXCLUDE_PATTERNS:
        if pattern.startswith('*'):
            if file_name.endswith(pattern[1:]):
                return True
        elif pattern.startswith('comparison_'):
            if file_name.startswith('comparison_') and file_name.endswith(('.mp4', '.gif')):
                return True
        elif file_name == pattern:
            return True
    
    return False

def should_exclude_dir(dir_path: Path) -> bool:
    """Check if a directory should be excluded."""
    dir_name = dir_path.name
    
    # Check exact matches
    if dir_name in EXCLUDE_DIRS:
        return True
    
    # Check if any parent is excluded
    for parent in dir_path.parents:
        if parent.name in EXCLUDE_DIRS:
            return True
    
    # Check path patterns (for subdirectories like demo/output)
    path_str = str(dir_path).replace('\\', '/')
    for pattern in EXCLUDE_PATH_PATTERNS:
        if pattern in path_str:
            return True
    
    return False

def collect_files(root_dir: Path):
    """Collect all files that should be included."""
    files_to_include = []
    files_to_exclude = []
    
    for root, dirs, files in os.walk(root_dir):
        root_path = Path(root)
        
        # Filter out excluded directories
        dirs[:] = [d for d in dirs if not should_exclude_dir(root_path / d)]
        
        for file in files:
            file_path = root_path / file
            
            if should_exclude_file(file_path):
                files_to_exclude.append(file_path)
            else:
                files_to_include.append(file_path)
    
    return files_to_include, files_to_exclude

def format_size(size_bytes):
    """Format file size in human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"

def main():
    root_dir = Path('.')
    
    print("=" * 70)
    print("PoseFormer Submission File Analyzer")
    print("=" * 70)
    print()
    
    files_to_include, files_to_exclude = collect_files(root_dir)
    
    # Calculate sizes
    total_size = sum(f.stat().st_size for f in files_to_include if f.exists())
    excluded_size = sum(f.stat().st_size for f in files_to_exclude if f.exists())
    
    print(f"[INCLUDE] Files to INCLUDE: {len(files_to_include)} files ({format_size(total_size)})")
    print(f"[EXCLUDE] Files to EXCLUDE: {len(files_to_exclude)} files ({format_size(excluded_size)})")
    print()
    
    # Show breakdown by directory
    print("Files to Include (by directory):")
    print("-" * 70)
    
    dir_counts = {}
    for file_path in files_to_include:
        # Get relative path from root
        rel_path = file_path.relative_to(root_dir)
        dir_name = str(rel_path.parent) if rel_path.parent != Path('.') else 'root'
        dir_counts[dir_name] = dir_counts.get(dir_name, 0) + 1
    
    for dir_name in sorted(dir_counts.keys()):
        print(f"  {dir_name}: {dir_counts[dir_name]} files")
    
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Total files to submit: {len(files_to_include)}")
    print(f"Total size: {format_size(total_size)}")
    print()
    
    if total_size > 100 * 1024 * 1024:  # 100MB
        print("WARNING: Submission size exceeds 100MB!")
        print("   Consider excluding more files or compressing.")
    else:
        print("Submission size looks good!")
    
    print()
    print("Tip: Review SUBMISSION_GUIDE.md for detailed instructions.")
    print()

if __name__ == '__main__':
    main()

