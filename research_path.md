some thoughts:
# Caching ground truth
1. given a prompt, extract the semantic meaning of the prompt and store it in a cache.
2. When result of a prompt is returned, store the ground truth in the cache.
3. When a new prompt is received, check the cache for similar ground truth. If found, used cached result, then token inference based on that. If not, run the prompt through the model and store the result in the cache.
4. The difference from a search engine reddis cache is that we only store ground truth, not the final result, the final result is generated based on the ground truth and the prompt.

# ditill a local gpt:
1. User interact with a edge mode, much smaller than cloud 
2. the edge model is a distilled version of the cloud model, but not as powerful. but only contain the proximity of infor the user is interested in:
3. For example, if the user asked alot of question about Stardew valley, the edge model will preemptively ask the cloud gpt to distill stardew valley related information and store it in the edge model.

