## Hotfixes

### HRNet Computational Graph Issue
Orginal issue: https://github.com/pytorch/pytorch/issues/30459

#### Reason

`None` in `nn.ModuleList` break the JIT in higher version of PyTorch. 

#### Solution

Taken from https://github.com/pytorch/pytorch/issues/30459#issuecomment-620462541

```python
class NoOpModule(nn.Module):
    """
    https://github.com/pytorch/pytorch/issues/30459#issuecomment-597679482
    """
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        return args
```

        
Then in make_transition_layer: `transition_layers.append(NoOpModule())`

And in _make_fuse_layers: `fuse_layer.append(NoOpModule())`

In forward, for each respective stage (e.g. stage 3 here):
```python
    if not isinstance(self.transition3[i], NoOpModule):
        x_list.append(self.transition3[i](y_list[-1]))
    else:
        x_list.append(y_list[i])
```

Note: need to change transition2 and transition3.