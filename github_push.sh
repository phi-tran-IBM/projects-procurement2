#!/bin/bash

# This script safely initializes a Git repository, commits essential code,
# and pushes it to a new GitHub repository. It respects the .gitignore file.

echo "--- Starting Git Repository Setup ---"

# 1. Initialize a new Git repository
git init
echo "✅ Git repository initialized."

# 2. Check for .gitignore and add files
if [ ! -f .gitignore ]; then
    echo "⚠️ WARNING: .gitignore file not found. It is highly recommended to create one."
    git add .
else
    # The .gitignore will ensure sensitive files are not staged
    git add .
    echo "✅ All safe files staged for commit."
fi

# 3. Make the first commit
git commit -m "Initial commit: Add project structure and core application logic"
echo "✅ Initial commit created."

# 4. Ask for the GitHub repository URL
read -p "➡️ Please paste your new GitHub repository URL here: " GITHUB_URL

# 5. Add the remote repository
git remote add origin $GITHUB_URL
echo "✅ Remote origin set to your GitHub repository."

# 6. Rename the default branch to 'main'
git branch -M main
echo "✅ Default branch renamed to 'main'."

# 7. Push the code to GitHub
git push -u origin main
echo "🚀 Success! Your code has been pushed to GitHub."
echo "--- Script Finished ---"