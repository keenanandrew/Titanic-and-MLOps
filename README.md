# Titanic-and-MLOps
MLFlow, Optuna and GitHub Actions experiments

The objective is to be able to create and deliver ML models.

In addition, these models should be:

- Robust
- Production-grade
- CI/CD enabled
- Documented
- Kitted out with common libraries:
- - MLFlow for tracking model iterations
- - Optuna for hyperparameter tuning


## Roadmap

- Update GitHub Actions to track things in Jira
- Create branches for feature requests straight from Jira

### Updates:

4 Dec 2023: 
- Added Jira integration
- Brought .gitignore up to standard
- CI/CD results now visible in Jira 


## Optuna

The optuna-experiments.py file is an example from MLEWP, using a random number generator. 
My goal is to take it apart, understand the key pieces, and understand how to implement Optuna in my titanic.py file.

- Need to research the MLFlow / Optuna connector