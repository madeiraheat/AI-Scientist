import openai
import os.path as osp
import shutil
import json
import argparse
import multiprocessing
import torch
import os
import time
import sys
import sympy
from aider.coders import Coder
from aider.models import Model
from aider.io import InputOutput
from datetime import datetime
from ai_scientist.generate_ideas import generate_ideas, check_idea_novelty
from ai_scientist.perform_experiments import perform_experiments
from ai_scientist.perform_writeup import perform_writeup, generate_latex
from ai_scientist.perform_review import perform_review, load_paper, perform_improvement

NUM_REFLECTIONS = 3

endpoint = os.getenv('ENDPOINT')
api_key = os.getenv('API_KEY')
deployment = os.getenv('DEPLOYMENT')

print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print(f"*Starting Review*")
writeup = "latex"

## REVIEW PAPER
if writeup == "latex":
    try:
        paper_text = load_paper("report.pdf")
        review = perform_review(
            paper_text,
            model="gpt-4o-2024-05-13",		    				 
            client=openai.AzureOpenAI(
                base_url=f"{endpoint}/openai/deployments/{deployment}/chat/completions?api_version=2024-02-15-preview",
                api_key=api_key, 
                api_version="2024-02-15-preview"
            ),
            num_reflections=5,
            num_fs_examples=1,
            num_reviews_ensemble=5,
            temperature=0.1,
        )
        # Store the review in separate review.txt file
        print(f"*Executed Perform_Review*")
        with open("review.txt", "w") as f:
            f.write(json.dumps(review, indent=4))
            print(f"*Dumped review into review.txt*")
    except Exception as e:
        print(f"Failed to perform review {e}")