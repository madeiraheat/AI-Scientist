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
print(f"*Improving Writeup*")
writeup = "latex"
improvement = True

paper_text = load_paper("report.pdf")
exp_file = "experiment.py"
vis_file = "plot.py"
notes = "notes.txt"
writeup_file = osp.join("", "latex", "template.tex")
#fnames = [paper_text, exp_file, writeup_file, notes]
fnames = [paper_text, notes]

io = InputOutput(
    yes=True, chat_history_file="history_aider.txt"
)
model="azure/gpt4o"
main_model = Model(model)
print("***Start Creating Coder***")
coder = Coder.create(
    main_model=main_model,
    fnames=fnames,
    io=io,
    stream=False,
    use_git=False,
    edit_format="diff",
)
print("***Coder created****")


## IMPROVE WRITEUP
if writeup == "latex" and improvement:
    print(f"*Starting Improvement*")
    with open("review.txt", "r", encoding='utf-8') as json_file:
        review = json.load(json_file)
        print(f"*Review.txt loaded*")
    try:
        #perform_improvement(review, coder)
        client_model = "gpt-4o-2024-05-13"
        client = openai.AzureOpenAI(
            base_url=f"{endpoint}/openai/deployments/{deployment}/chat/completions?api_version=2024-02-15-preview",
            api_key=api_key, 
            api_version="2024-02-15-preview"
        )

        perform_writeup("DUALSCALE DIFFUSION: ADAPTIVE FEATURE BALANCING FOR LOW-DIMENSIONAL GENERATIVE MODELS", "", coder, client, client_model)
    except Exception as e:
        print(f"Failed to perform writeup: {e}")
       
        print("Done writeup")

        print(f"*Executed perform_improvement*")
        generate_latex(coder,"", "improvedpaper.pdf")
        print(f"*Generated Improvedpaper.pdf*")
        paper_text = load_paper("improvedpaper.pdf")
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
        print(f"*Executed perform_review over Improvedpaper.pdf*")
        # Store the review in separate review.txt file
        with open("review_improved.txt", "w") as f:
            f.write(json.dumps(review))
            print(f"*Dumped review over Improvedpaper.pdf into review_improved.txt*")
    except Exception as e:
            print(f"Failed to perform improvement: {e}")