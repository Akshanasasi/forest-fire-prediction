# GitHub Setup Instructions

Follow these steps to push your Wildfire Prediction App to GitHub.

## Before You Start

1. **Create a GitHub account** if you don't have one: https://github.com/join
2. **Create a new repository** on GitHub:
   - Go to https://github.com/new
   - Name it something like `wildfire-prediction-app`
   - Keep it public or private (your choice)
   - Do NOT initialize with README, .gitignore, or license
   - Click "Create repository"

## Step-by-Step Setup

### Method 1: Edit and Run the Script (Easiest)

1. Open the file `setup_github.sh` in this project
2. Replace the following placeholders with your information:
   - `YOUR_GITHUB_USERNAME` → Your actual GitHub username
   - `your.email@example.com` → Your email associated with GitHub
   - `YOUR_REPO_NAME` → The name of your repository (e.g., `wildfire-prediction-app`)

3. Run the script in the Shell:
   ```bash
   chmod +x setup_github.sh
   ./setup_github.sh
   ```

### Method 2: Manual Commands

Run these commands one by one in the Shell, replacing the placeholders:

```bash
# Configure Git with your information
git config --global user.name "YOUR_GITHUB_USERNAME"
git config --global user.email "your.email@example.com"

# Initialize repository
git init

# Add all files
git add .

# Create first commit
git commit -m "Initial commit - Wildfire prediction app"

# Connect to GitHub (replace YOUR_GITHUB_USERNAME and YOUR_REPO_NAME)
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPO_NAME.git

# Push to GitHub
git branch -M main
git push -u origin main
```

## Authentication

When you run `git push`, you'll be asked for authentication:
- **Username**: Your GitHub username
- **Password**: Use a Personal Access Token (NOT your GitHub password)

### How to Create a Personal Access Token:
1. Go to https://github.com/settings/tokens
2. Click "Generate new token" → "Generate new token (classic)"
3. Give it a name like "Replit Access"
4. Select scopes: Check `repo` (full control of private repositories)
5. Click "Generate token"
6. **Copy the token immediately** (you won't see it again!)
7. Use this token as your password when pushing to GitHub

## Example with Real Values

If your GitHub username is `johndoe` and your repo is `wildfire-app`:

```bash
git config --global user.name "johndoe"
git config --global user.email "john.doe@gmail.com"
git remote add origin https://github.com/johndoe/wildfire-app.git
```

## Troubleshooting

- **"remote origin already exists"**: Run `git remote remove origin` first
- **Authentication failed**: Make sure you're using a Personal Access Token, not your password
- **Nothing to commit**: Your changes might already be committed

## After Pushing

Your code will be available at:
`https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPO_NAME`

Share this link with anyone to show them your project!
