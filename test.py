import torch
from pprint import pprint

'''
mat_a = torch.tensor(
    [
        [
            [1, 2],
            [2, 1],
            [0, 1]
        ],
        [
            [3, 4],
            [4, 3],
            [5, 5]
        ],
        [
            [3, 4],
            [4, 3],
            [5, 5]
        ],
        [
            [3, 4],
            [4, 3],
            [5, 5]
        ]
    ]
)
pprint(mat_a.shape)
pprint(mat_a)


mat_W = torch.randn((3, 5), dtype=torch.float64)
mat_X = torch.randn((10, 3), dtype=torch.float64)
mat_c = mat_W @ mat_X.T
mat_d = mat_X @ mat_W
pprint(mat_c.shape)
pprint(mat_c)
pprint(mat_d.shape)
pprint(mat_d)
'''
a = torch.tensor([
    [1, 2, 3],
    [1, 2, 3]
])

b = torch.tensor([5, 6, 7])
pprint(b + a)