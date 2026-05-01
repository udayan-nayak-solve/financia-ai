#!/bin/bash
# Setup Git LFS for large data files

echo "🔧 Setting up Git LFS for large data files..."

# Check if git-lfs is installed
if ! command -v git-lfs &> /dev/null; then
    echo "📦 Installing Git LFS..."
    brew install git-lfs
fi

# Initialize Git LFS
git lfs install

# Track large CSV files in data/actual
git lfs track "data/actual/*.csv"

# Add .gitattributes
git add .gitattributes

echo "✅ Git LFS setup complete!"
echo ""
echo "📊 Files that will be tracked by LFS:"
echo "   - data/actual/*.csv"
echo ""
echo "Next steps:"
echo "1. git add data/actual/"
echo "2. git commit -m 'Add data/actual files with Git LFS'"
echo "3. git push origin main"
echo ""
echo "⚠️  Note: GitHub Free has 1GB LFS storage limit per month"
echo "   Your data/actual is 294MB, which fits within the limit"
