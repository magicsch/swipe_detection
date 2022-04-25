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

Which shows a debug window which tells us what the program is detecting.
Read main.py for example on how to use Swipe Classifier

## Swipe Classifier

This class deals with loading model and config and detecting the swipes. It ouputs the edges of the swipe signal. Meaning that the first Swipe.RIght indicates the swipe has started and the second indicates it has ended. So when the model outputs the second Swipe.Right a right swipe has been performed.

![alt text](https://github.com/man-do/swipe_detection/blob/main/imgs/flow.jpg)

## Exit

Press ctrl+c in terminal to exit. or when with debug window pres q.
