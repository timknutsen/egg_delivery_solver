#!/bin/bash
# Run from: /Users/tikn/Desktop/code/egg_delivery_solver

cd /Users/tikn/Desktop/code/egg_delivery_solver

# Stage the renamed app.py and new files
git add new_version_for_meeting/app.py
git add new_version_for_meeting/requirements.txt
git add .gitignore
git add README.md

# Commit everything
git commit -m "Clean up: single app.py, remove old versions, add .gitignore"

# Push
git push origin main

echo "✅ Git updated!"
