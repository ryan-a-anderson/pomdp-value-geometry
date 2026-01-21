# Quick Start Guide

Get your POMDP value function geometry code on GitHub in 5 minutes!

## Step 1: Create GitHub Repository (2 minutes)

1. Go to https://github.com/new
2. Fill in:
   - **Repository name**: `pomdp-value-geometry`
   - **Description**: `Value Function Geometry for Memoryless POMDPs - Code for research paper`
   - **Visibility**: Public ✓ (or Private if not ready to share)
3. **IMPORTANT**: Leave these UNCHECKED:
   - [ ] Add a README file
   - [ ] Add .gitignore
   - [ ] Choose a license
4. Click "Create repository"

## Step 2: Connect and Push (1 minute)

Copy and run these commands in your terminal (replace `YOUR_USERNAME`):

```bash
cd "/Users/ryananderson/Documents/Programming/Value Function Geometry/new_experiments_jan_20"

git remote add origin https://github.com/YOUR_USERNAME/pomdp-value-geometry.git

git push -u origin main
```

**That's it!** Your code is now on GitHub! 🎉

## Step 3: Make README Visible (1 minute)

So GitHub shows the nice README on your repository page:

```bash
git mv README.md README_TECHNICAL.md
git mv README_GITHUB.md README.md
git commit -m "Update README for GitHub display"
git push
```

## Step 4: Configure Repository (1 minute)

On your GitHub repository page:

1. Click the ⚙️ (gear icon) next to "About"
2. Add topics (helps people find your repo):
   - `pomdp`
   - `reinforcement-learning`
   - `optimization`
   - `value-functions`
   - `markov-decision-process`
3. Click "Save changes"

## Done!

Your repository is now:
- ✅ Live on GitHub
- ✅ Properly documented
- ✅ Ready to share
- ✅ Prepared for paper citation

## What's Next?

### For Paper Publication

When ready to publish your paper:

```bash
# Create a release version
git tag -a v1.0.0 -m "Code release for paper publication"
git push origin v1.0.0

# Then on GitHub:
# Releases → Create a new release → Choose tag v1.0.0
# Add release notes and attach paper PDF
```

### Get a DOI (Permanent Citation)

1. Go to https://zenodo.org
2. Sign in with GitHub
3. Enable your repository in Zenodo settings
4. Create a release on GitHub → Zenodo auto-generates DOI
5. Add DOI badge to README

### Share Your Work

Now you can:
- Share the GitHub link in your paper
- Tweet/post about your research with the repo link
- Add to your CV/website
- Accept contributions from other researchers

## Troubleshooting

### Authentication Failed?

**Option 1**: Use Personal Access Token
1. GitHub → Settings → Developer settings → Personal access tokens
2. Generate new token (classic) with `repo` scope
3. Use token as password when pushing

**Option 2**: Use SSH
```bash
# Generate SSH key
ssh-keygen -t ed25519 -C "your-email@example.com"

# Start ssh-agent and add key
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519

# Copy public key
cat ~/.ssh/id_ed25519.pub
# Add to GitHub: Settings → SSH and GPG keys

# Use SSH remote instead
git remote set-url origin git@github.com:YOUR_USERNAME/pomdp-value-geometry.git
```

### Already Have 'origin' Remote?

```bash
# Check current remotes
git remote -v

# Remove old origin
git remote remove origin

# Add new one
git remote add origin https://github.com/YOUR_USERNAME/pomdp-value-geometry.git
```

### Need to Undo Something?

```bash
# Undo last commit (keep changes)
git reset --soft HEAD~1

# Undo all changes to a file
git checkout -- filename

# See what changed
git diff
```

## Key Commands Reference

```bash
# Check status
git status

# See commits
git log --oneline

# Make changes
git add .
git commit -m "Your message"
git push

# Get latest from GitHub
git pull

# Create branch
git checkout -b feature-name
```

## Need More Help?

- **Detailed setup**: See `GITHUB_SETUP.md`
- **Contributing**: See `CONTRIBUTING.md`
- **Full summary**: See `REPOSITORY_SUMMARY.txt`
- **GitHub guides**: https://guides.github.com

## Repository URL

After pushing, your code will be at:
```
https://github.com/YOUR_USERNAME/pomdp-value-geometry
```

Share this link in your paper, CV, and with collaborators!
