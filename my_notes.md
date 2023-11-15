# Made with ML Notes

My notes for Made with ML course.

## Design

**Setup**: Clone git repo and move files to my repo:

```bash
export GITHUB_USERNAME="YOUR_GITHUB_UESRNAME"  # <-- CHANGE THIS to your username
git clone https://github.com/GokuMohandas/Made-With-ML.git .
git remote set-url origin https://github.com/$GITHUB_USERNAME/Made-With-ML.git
git checkout -b dev
export PYTHONPATH=$PYTHONPATH:$PWD  # so we can import modules from our scripts
```

**ML Product Design and System Design Development**. See https://madewithml.com/static/templates/ml-canvas.pdf

* Product Design (What and why?): https://madewithml.com/courses/mlops/product-design/
* System Design (How?): https://madewithml.com/courses/mlops/systems-design/

Address data imbalance: There are many strategies, including over-sampling less frequent classes and under-sampling popular classes, class weights in the loss function, etc.

## Model

**Model optimization after training** :

**Pruning**: remove weights (unstructured) or entire channels (structured) to reduce the size of the network. The objective is to preserve the model’s performance while increasing its sparsity.

**Quantization**: reduce the memory footprint of the weights by reducing their precision (ex. 32 bit to 8 bit). We may loose some precision but it shouldn’t affect performance too much.

**Distillation**: training smaller networks to “mimic” larger networks by having it reproduce the larger network’s layers’ outputs.

Also there is ONNX.

**Experiment tracking**: MLFlow https://mlflow.org/ or W&B https://wandb.ai/site/

**Experiment tuning**: Ray tune, Optuna, Hyperopt, etc.

**Experiment evaluation**: Relevant metrics

Behavioral testing:

Besides just looking at metrics, we also want to conduct some behavioral sanity tests. Behavioral testing is the process of testing input data and expected outputs while treating the model as a black box. They don't necessarily have to be adversarial in nature but more along the types of perturbations we'll see in the real world once our model is deployed. A landmark paper on this topic is Beyond Accuracy: Behavioral Testing of NLP Models with CheckList which breaks down behavioral testing into three types of tests: invariance, directional, minimum functionality.

Online evaluation: 
- manually label a subset of incoming data to evaluate periodically.
- asking the initial set of users viewing a newly categorized content if it's correctly classified.
- allow users to report misclassified content by our model.

AB tests: AB testing involves sending production traffic to our current system (control group) and the new version (treatment group) and measuring if there is a statistical difference between the values for two metrics. There are several common issues with AB testing such as accounting for different sources of bias, such as the novelty effect of showing some users the new system. We also need to ensure that the same users continue to interact with the same systems so we can compare the results without contamination. In many cases, if we're simply trying to compare the different versions for a certain metric, AB testing can take while before we reach statical significance since traffic is evenly split between the different groups. In this scenario, multi-armed bandits will be a better approach since they continuously assign traffic to the better performing version.

Canary tests: Canary tests involve sending most of the production traffic to the currently deployed system but sending traffic from a small cohort of users to the new system we're trying to evaluate. Again we need to make sure that the same users continue to interact with the same system as we gradually roll out the new system.

Shadow tests: Shadow testing involves sending the same production traffic to the different systems. We don't have to worry about system contamination and it's very safe compared to the previous approaches since the new system's results are not served. However, we do need to ensure that we're replicating as much of the production system as possible so we can catch issues that are unique to production early on. But overall, shadow testing is easy to monitor, validate operational consistency, etc.

Capability vs. alignment: We've seen the many different metrics that we'll want to calculate when it comes to evaluating our model but not all metrics mean the same thing. And this becomes very important when it comes to choosing the "best" model(s).

* capability: the ability of our model to perform a task, measured by the objective function we optimize for (ex. log loss)
* alignment: desired behavior of our model, measure by metrics that are not differentiable or don't account for misclassifications and probability differences (ex. accuracy, precision, recall, etc.)

While capability (ex. loss) and alignment (ex. accuracy) metrics may seem to be aligned, their differences can indicate issues in our data:

* ↓ accuracy, ↑ loss = large errors on lots of data (worst case)
* ↓ accuracy, ↓ loss = small errors on lots of data, distributions are close but tipped towards misclassifications (misaligned)
* ↑ accuracy, ↑ loss = large errors on some data (incorrect predictions have very skewed distributions)
* ↑ accuracy, ↓ loss = no/few errors on some data (best case)*

**Model serving**:

There are many frameworks to choose from when it comes to model serving, such as Ray Serve, Nvidia Triton, HuggingFace, Bento ML, etc.

Types: batch inference (offline) and online inference (real-time)

## Developing

Move from notebooks to scripts.

Execute scripts from terminal.

Note: Place the main function call under a `if __name__ == "__main__"` conditional so that it's only executed when we run the script directly. Here we can pass in the input arguments directly into the function in the code.

```python
# madewithml/train.py
if __name__ == "__main__":
    train_model(experiment_name="llm", ...)
```

```bash
python madewithml/train.py
```

Use argparse:

```python
# madewithml/serve.py
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", help="run ID to use for serving.")
    parser.add_argument("--threshold", type=float, default=0.9, help="threshold for `other` class.")
    args = parser.parse_args()
    ray.init()
    serve.run(ModelDeployment.bind(run_id=args.run_id, threshold=args.threshold))
```

```bash
python madewithml/serve.py --run_id $RUN_ID
```

Typer: https://typer.tiangolo.com/

## Utilities

Logging: see below.

```python
import logging
import sys

# Create super basic logger
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

# Logging levels (from lowest to highest priority)
logging.debug("Used for debugging your code.")
logging.info("Informative messages from your code.")
logging.warning("Everything works but there is something to be aware of.")
logging.error("There's been a mistake with the process.")
logging.critical("There is something terribly wrong and process may terminate.")
```

Create documentations: https://madewithml.com/courses/mlops/documentation/

Style and format code: see below.

- Black: an in-place reformatter that (mostly) adheres to PEP8
- isort: sorts and formats import statements inside Python scripts.
- flake8: a code linter with stylistic conventions that adhere to PEP8.

Clean way to do it is using .toml file.

```toml
# pyproject.toml

# Black formatting
[tool.black]
line-length = 150
include = '\.pyi?$'
exclude = '''
/(
      .eggs         # exclude a few common directories in the
    | .git          # root of the project
    | .hg
    | .mypy_cache
    | .tox
    | venv
    | _build
    | buck-out
    | build
    | dist
  )/
'''

# iSort
[tool.isort]
profile = "black"
line_length = 79
multi_line_output = 3
include_trailing_comma = true
virtual_env = "venv"

[tool.flake8]
exclude = "venv"
ignore = ["E501", "W503", "E226", "E266"]
# E501: Line too long
# W503: Line break occurred before binary operator
# E226: Missing white space around arithmetic operator
# E266: Too many leading '#' for block comment

[tool.pyupgrade]
py39plus = true
```

To use it, run:

```bash
black .
flake8
isort .
```

```
# Makefile
SHELL = /bin/bash

# Styling
.PHONY: style
style:
    black .
    flake8
    python3 -m isort .
    pyupgrade

# Cleaning
.PHONY: clean
clean: style
    find . -type f -name "*.DS_Store" -ls -delete
    find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
    find . | grep -E ".pytest_cache" | xargs rm -rf
    find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
    rm -rf .coverage*
```

Pre-commit hooks: https://madewithml.com/courses/mlops/pre-commit/

```yaml
# .pre-commit-config.yaml

# See https://pre-commit.com for more information
# See https://pre-commit.com/hooks.html for more hooks
repos:
-   repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
    -   id: trailing-whitespace
    -   id: end-of-file-fixer
    -   id: check-merge-conflict
    -   id: check-yaml
    -   id: check-added-large-files
        args: ['--maxkb=1000']
        exclude: "notebooks"
    -   id: check-yaml
        exclude: "mkdocs.yml"
-   repo: local
    hooks:
    -   id: clean
        name: clean
        entry: make
        args: ["clean"]
        language: system
        pass_filenames: false
```

## Testing

**Code**: Test ML artifacts.

Types of test: https://madewithml.com/courses/mlops/testing/

* Unit tests: tests on individual components that each have a single responsibility (ex. function that filters a list).
* Integration tests: tests on the combined functionality of individual components (ex. data processing).
* System tests: tests on the design of a system for expected outputs given inputs (ex. training, inference, etc.).
* Acceptance tests: tests to verify that requirements have been met, usually referred to as User Acceptance Testing (UAT).
* Regression tests: tests based on errors we've seen before to ensure new changes don't reintroduce them.

The framework to use when composing tests is the Arrange Act Assert methodology.
- Arrange: set up the different inputs to test on.
- Act: apply the inputs on the component we want to test.
- Assert: confirm that we received the expected output.

### CI/CD for Machine Learning

CI/CD workflows using GitHub Actions.

### Monitoring Machine Learning Systems

Need drift detection: https://madewithml.com/courses/mlops/monitoring/#drift

Monitor: https://github.com/GokuMohandas/monitoring-ml

#### Drift

**Data drift**: Data drift, also known as feature drift or covariate shift, occurs when the distribution of the production data is different from the training data. The model is not equipped to deal with this drift in the feature space and so, it's predictions may not be reliable. The actual cause of drift can be attributed to natural changes in the real-world but also to systemic issues such as missing data, pipeline errors, schema changes, etc. It's important to inspect the drifted data and trace it back along it's pipeline to identify when and where the drift was introduced.

**Target drift**: Besides just the input data changing, as with data drift, we can also experience drift in our outcomes. This can be a shift in the distributions but also the removal or addition of new classes with categorical tasks. Though retraining can mitigate the performance decay caused target drift, it can often be avoided with proper inter-pipeline communication about new classes, schema changes, etc.

**Concept drift**: Besides the input and output data drifting, we can have the actual relationship between them drift as well. This concept drift renders our model ineffective because the patterns it learned to map between the original inputs and outputs are no longer relevant. Concept drift can be something that occurs in various patterns.

#### Monitoring

When it actually comes to implementing a monitoring system, we have several options, ranging from fully managed to from-scratch. Several popular managed solutions are Arize, Arthur, Fiddler, Gantry, Mona, WhyLabs, etc., all of which allow us to create custom monitoring views, trigger alerts, etc. There are even several great open-source solutions such as EvidentlyAI, TorchDrift, WhyLogs, etc.

We'll often notice that monitoring solutions are offered as part of the larger deployment option such as Sagemaker, TensorFlow Extended (TFX), TorchServe, etc. And if we're already working with Kubernetes, we could use KNative or Kubeless for serverless workload management. But we could also use a higher level framework such as KFServing or Seldon core that natively use a serverless framework like KNative.

