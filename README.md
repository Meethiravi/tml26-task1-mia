### TML2026 Task 1 - Membership Attack

This repo contains the submission for our assignment for TML Task 1. 

# Clone the repository

    git clone https://github.com/Meethiravi/tml26-task1-mia.git

# Prepare Files

wget https://huggingface.co/datasets/SprintML/tml26_task1/resolve/main/pub.pt
wget https://huggingface.co/datasets/SprintML/tml26_task1/resolve/main/priv.pt
wget https://huggingface.co/datasets/SprintML/tml26_task1/resolve/main/model.pt

# Install the dependencies

    pip install -r requirements.txt

# Replace API_KEY

    In task_template.py replace "YOUR_API_KEY" with your actual API Key.

# Submit 
    condor_submit mia.sub 
 






