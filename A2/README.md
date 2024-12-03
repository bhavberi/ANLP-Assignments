# ANLP A-3

> **Name**: Bhav Beri
>
> **Roll Number**: 2021111013

----

This repository contains a report for the Assignment 3 of the Advanced Natural Language Processing (ANLP) course. The report discusses the key concepts of transformer architecture, self-attention, positional encodings, and hyperparameter tuning for machine translation tasks.

## Running the Code

Link to Saved Models, which need to be put in the `models` directory to run inferences: [https://iiitaphyd-my.sharepoint.com/:f:/g/personal/bhav_beri_research_iiit_ac_in/EtMFF4Sn6jtGhshk9nMBb4sBe9l-UbK2-65CGT2hJetBnA?e=DMfhhw](https://iiitaphyd-my.sharepoint.com/:f:/g/personal/bhav_beri_research_iiit_ac_in/EtMFF4Sn6jtGhshk9nMBb4sBe9l-UbK2-65CGT2hJetBnA?e=DMfhhw)

- To run the code, firstly install the required packages from pip. The packages in the environment used to train the models can be found in the [freeze file](./code/freeze_file.txt).
- Make sure to have the saved models in the `models` directory.
- `cd code`
- Run the `test.py` file to get the BLEU score on the test data. The results would be saved at `../results` directory.

## Report
The Report can be found [here (in md format)](./report.md) and [here (in pdf format)](./report.pdf).

## Results
The main results for the best performing model can be found in the folder [results_heads4](./results_heads4/), while the results for the second best performance can be found in the folder [results_heads6](./results_heads6/) (The two models differ in the number of attention heads used).