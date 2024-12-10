# For just the specific example
import torch

def func(x):
    return x**2

jac_fn = torch.func.jacfwd(func)
batched_jac_fn = torch.vmap(jac_fn)
batched_input = torch.tensor([[1.0], [2.0], [3.0]])

with torch.inference_mode():
    result = batched_jac_fn(batched_input)  # This will raise the error
