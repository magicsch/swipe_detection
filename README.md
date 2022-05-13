# Swipe detection

Detect 4 types of swipes used for human machine interaction.

- RIGHT
- LEFT
- UP
- DOWN

# Installing Python

Install python from https://www.python.org/downloads/release/python-3911/

# Create Virtual Environment(Optional)

## Install the virtualenv tool

`pip install virtualenv`

## Create Project for Virtualenvironment(Optional)

` python -m pip install virtualenv`

## Activate Virtual Environment(Optional)

`source myproject/venv/bin/activate `

## Now clone Repo and install requirments(Required)

## Clone the repo

`git clone https://github.com/man-do/swipe_detection` OR Download this repo and unzip with winrar

## Move to repo local directory

`cd swipe_detection`

## Install dependencies

`pip install -r requirements.txt`

## Run

`python main.py`

Runs an example of the program with a debug window. Read the script to know how to use the program.

## Swipe Classifier

This class deals with loading model and config and detecting the swipes.

![alt text](https://github.com/man-do/swipe_detection/blob/main/imgs/flow.jpg)

## Exit

Press ctrl+c in terminal to exit. or when with debug window pres q.
