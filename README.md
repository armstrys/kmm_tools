# Kaggle March Madness Tools

This repository has useful tools that can also be found on kaggle


## Bracket Interface

Set an environment variable on your machine for the competition data path location. The easiest way to do this is by making a file called .env in this folder with the following content

```
COMPETITION_DATA_PATH = "/Absoulte/path/to/competition/data"
```

Then install `uv` and run:

```bash
uv run streamlit run kmm_tools/march_madness_interface.py
```