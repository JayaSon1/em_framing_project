# Create virtual environment
python -m venv .venv

# Activating virtual environment
.venv\Scripts\Activate.ps1

# Packages
pip install torch transformers peft trl accelerate pyyaml pandas openai python-dotenv

# Git
# Initialise git to start a fresh tracking history: 
git init

# Stage files so git prepares them for upload
git add .

# Commit files locally
git commit -m "Initial commit"

# Rename default branch to main
git branch -M main

# Link repository URL
git remote add origin https://...
 
# Push code to github
git push -u origin main

# Clone Turner datasets from github
git clone --depth 1 https://github.com/clarifying-EM/model-organisms-for-EM.git
pip install easy-dataset-share
cd model-organisms-for-EM/em_organism_dir/data
easy-dataset-share unprotect-dir training_datasets.zip.enc -p model-organisms-em-datasets --remove-canaries

# To run generate
python generate.py        # free, ~45 min per model
python judge.py --dry-run # how many API calls, no cost
python judge.py           # the paid part