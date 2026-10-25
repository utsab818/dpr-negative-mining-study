# Exploring Dense Passage Retrieval

I have been trying to understand how retrieval models work beyond just using them. For this project, I picked Dense Passage Retrieval (DPR) and explored how it is trained and how different training choices affect the passages it retrieves.

The main thing I was curious about was negative passages. A retriever needs to learn which passage is relevant to a question, but it also needs examples of passages that are not relevant. I wanted to see what changes when those negative examples come from the same training batch or when we use harder examples that look more relevant.

## What I explored

I experimented with in-batch negatives and hard negatives during DPR training. I also looked at training parameters and how changing them affects the model and its retrieval results.

Instead of looking only at the training process, I checked the passages returned for questions. This helped me connect what was happening during training with what the retriever was actually finding.

Some of the questions I tried to understand were

1. How does DPR learn to place questions and relevant passages close to each other?
2. What is the difference between using in-batch negatives and hard negatives?
3. How do training choices affect which passages appear in the retrieval results?
4. What can we learn by looking at passages that were retrieved but were not relevant?

## What I learned

One thing that stood out to me is that choosing negative examples is an important part of training a retriever. In-batch negatives make use of passages already present in a batch, while hard negatives give the model more difficult examples to distinguish from the correct passage.

I also realized that looking at retrieved passages is useful alongside training settings and metrics. It gives a better idea of where the model is doing well and where it still gets confused.

## Final thoughts

This project was mainly about understanding DPR by experimenting with it myself. I did not try every possible setup, and there are still many things I would like to test. But it helped me understand the role of negatives, training parameters, and passage retrieval a little better.

I have also written a blog about this exploration where I go through the ideas and experiments in more detail.

https://hackmd.io/c-zg2nPIQ2SCYh3JDASjNQ