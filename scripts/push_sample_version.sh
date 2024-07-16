#!/bin/bash
# Fail the script if any command fails
set -e

# Extract version from txt file
version=$(cat configs/data_version.txt)
echo "Version: $version"

# Commit file version to Git
echo "Committing file version to Git"
git add data/samples/
git add configs/
git commit -m "chore: update sample data $version" --no-verify

# Add tag with the version
echo "Adding tag with the version"
git tag -a $version -m "version $version"

# Push changes to Git
echo "Pushing changes to Git"
git push origin main --tags
