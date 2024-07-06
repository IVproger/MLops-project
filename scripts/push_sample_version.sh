#!/bin/bash
# Fail the script if any command fails
set -e

# Extract version from txt file
version=$(cat configs/data_version.txt)
echo "Version: $version"

# Add file to DVC
echo "Adding file to DVC"
dvc add data/samples/sample.csv
dvc push

# Commit file version to Git
echo "Committing file version to Git"
git add data/samples/
git add configs/
git commit -m "chore: update sample data $version"

# Add tag with the version
echo "Adding tag with the version"
git tag -a $version -m "version $version"

# Push changes to Git
echo
read -p "Do you want to push changes to Git? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
  echo "Pushing changes to Git"
  git push origin main --tags
else
  echo "Changes not pushed to Git. Push them with 'git push origin main --tags'"
fi
