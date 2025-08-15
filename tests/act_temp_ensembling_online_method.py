import torch

seq = torch.linspace(8, 8.5, 100)
print(seq)

m = 0.01
exp_weights = torch.exp(-m * torch.arange(len(seq)))
print(exp_weights)

# Calculate offline
avg = (exp_weights * seq).sum() / exp_weights.sum()
print("offline", avg)

# Calculate online
for i, item in enumerate(seq):
	if i == 0:
		avg = item
		continue
	avg *= exp_weights[:i].sum()
	avg += item * exp_weights[i]
	avg /= exp_weights[:i+1].sum()
print("online", avg)
