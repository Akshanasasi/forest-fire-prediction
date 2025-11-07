#!/bin/bash

# GitHub Setup Script for @janarajan04
# 
# BEFORE RUNNING THIS SCRIPT:
# 1. Create a new repository on GitHub at: https://github.com/new
# 2. Name it (e.g., wildfire-prediction-app)
# 3. Leave it empty (don't initialize with README)
#
# THEN EDIT THESE TWO LINES BELOW:
# - Replace YOUR_EMAIL with your email address
# - Replace YOUR_REPO_NAME with the name you gave your repository

# STEP 1: Configure your Git identity
git config --global user.name "janarajan04"
git config --global user.email "YOUR_EMAIL@example.com"

# STEP 2: Initialize Git repository (if not already done)
git init

# STEP 3: Add all files to Git
git add .

# STEP 4: Create your first commit
git commit -m "Initial commit - Wildfire prediction app"

# STEP 5: Connect to your GitHub repository
# Replace YOUR_REPO_NAME with your actual repository name
# Example: If your repo is https://github.com/janarajan04/wildfire-app
# Then change YOUR_REPO_NAME to: wildfire-app
git remote add origin https://github.com/janarajan04/YOUR_REPO_NAME.git

# STEP 6: Push to GitHub
# You will be prompted for your GitHub username and Personal Access Token
git branch -M main
git push -u origin main

echo "Setup complete! Your code should now be on GitHub at:"
echo "https://github.com/janarajan04/YOUR_REPO_NAME"
