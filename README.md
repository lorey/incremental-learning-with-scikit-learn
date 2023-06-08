# Incremental Learning with scikit-learn

This repo implements incremental learning with scikit-learn
by using a sample dataset containing a concept drift,
i.e. a target variable that changes its meaning over time.

### Scenario

Users choose items they like. While they reject everything in the beginning `f(g)=0`,
a regular pattern emerges over time `f(g)=-x`.

I used this sample to understand the effects of `partial_fit`
with respect to both fitting time and quality of the training.

### Methods

- `partial_all`: partial_fit on all data until this point
- `partial_step`: partial_fit on batch data
- `full_step`: regular fit on batch data (ignores all previous samples!)
- `full_all`: regular fit on all data until this point

PS: Please be aware that I have no idea what I'm doing.
