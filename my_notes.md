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
