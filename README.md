# Incremental Learning with scikit-learn

This repo implements incremental learning with scikit-learn
by using a sample dataset containing a concept drift,
i.e. a target variable that changes its meaning over time.

I used it to understand the effects of `partial_fit`
with respect to both fitting time and quality.

Summary:

- doing a partial_fit on all known data over time converges quickly and is fast (`partial_all`).
- doing a partial_fit on each batch instead of all known data is quickest, but slow to converge (`partial_step`).
- fitting on all known samples all the time obviously yields the best results but is computationally 5x more expensive than continuing training with a partial fit on all data (`full_all`)
- doing a full fit for each step individually does not make sense as you would only train on most recent data only and throw away previous examples. Interestingly, this takes almost as long as fitting on all data each time (`full_step`)

PS: Please be aware that I have no idea what I'm doing.
