# Setting Up GitHub Repository

Follow these steps to create and push your PoseFormer project to GitHub.

## Step 1: Create Repository on GitHub

1. Go to [GitHub.com](https://github.com) and sign in
2. Click the **+** icon in the top right → **New repository**
3. Repository name: `PoseFormer`
4. Description: "Fitness coaching system using 3D human pose estimation"
5. Choose **Public** or **Private**
6. **DO NOT** initialize with README, .gitignore, or license (we already have these)
7. Click **Create repository**

## Step 2: Initialize Git (if not already done)

Open PowerShell/Terminal in your project directory:

```powershell
cd C:\Users\rumiq\Desktop\PoseFormerV2

# Initialize git (if not already done)
git init

# Add all files
git add .

# Make initial commit
git commit -m "Initial commit: Fitness coaching system with PoseFormerV2"
```

## Step 3: Connect to GitHub and Push

```powershell
# Add remote repository (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/PoseFormer.git

# Rename branch to main (if needed)
git branch -M main

# Push to GitHub
git push -u origin main
```

## Step 4: Update README (Optional)

You can rename `README_FITNESS.md` to `README.md` if you want it as the main README:

```powershell
# Backup original README
mv README.md README_ORIGINAL.md

# Use fitness README as main
mv README_FITNESS.md README.md
```

## Alternative: Using GitHub CLI

If you have GitHub CLI installed:

```powershell
# Create repository and push in one command
gh repo create PoseFormer --public --source=. --remote=origin --push
```

## What Gets Uploaded

The `.gitignore` file ensures these are **NOT** uploaded:
- Cached videos (`user_videos_cache/`)
- Processed references (`references/`)
- Large model files (`checkpoint/`, `*.pth`, `*.bin`)
- Video files (`*.mp4`, `*.gif`)
- Python cache files (`__pycache__/`, `*.pyc`)

These **ARE** uploaded:
- All source code (`fitness_coach/`, `demo/`, etc.)
- Configuration files (`requirements.txt`, etc.)
- Documentation (`README.md`, etc.)

## After Pushing

1. Go to your repository on GitHub
2. Add a description and topics (e.g., `pose-estimation`, `fitness`, `computer-vision`, `pytorch`)
3. Consider adding a license file (MIT, Apache 2.0, etc.)
4. Update the README with any additional information

## Troubleshooting

### Authentication Issues

If you get authentication errors, use a Personal Access Token:

1. GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Generate new token with `repo` permissions
3. Use token as password when pushing

### Large Files

If you accidentally try to push large files, Git will warn you. Remove them:

```powershell
git rm --cached large_file.mp4
git commit -m "Remove large file"
```

