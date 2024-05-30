import torch
from pprint import pprint

"""
https://pytorch.org/docs/stable/generated/torch.matmul.html
Best broadcast explanation
https://deeplearninguniversity.com/pytorch/pytorch-broadcasting/#:~:text=Broadcasting%20functionality%20in%20Pytorch%20has,if%20certain%20constraints%20are%20met.
"""

"""
If both tensors are 1-dimensional, the dot product (scalar) is returned
"""
pprint('1.')
a = torch.tensor([1, 2, 3])
b = torch.tensor([2, 3, 4])
pprint(a @ b)

"""
If both arguments are 2-dimensional, the matrix-matrix product is returned.
"""
pprint('2.')
a = torch.tensor(
    [
        [1, 2, 3],
        [4, 5, 6]
    ]
)
b = torch.tensor(
    [
        [1, 2],
        [3, 4],
        [5, 6]
    ]
)
"""
1*1 + 2*3 + 3*5 = 22    1*2 + 2*4 + 3*6 = 28
4*1 + 5*3 + 6*5 = 49    4*2 + 5*4 + 6*6 = 64 
"""
c = a @ b
pprint(c.shape)
pprint(c)
"""
Multidimensional option
"""
pprint('3.')
a = torch.tensor(
    [
        [
            [
                [1, 2, 3],
                [4, 5, 6]
            ],
            [
                [1, 2, 3],
                [4, 5, 6]
            ]
        ],
        [
            [
                [1, 2, 3],
                [4, 5, 6]
            ],
            [
                [1, 2, 3],
                [4, 5, 6]
            ]
        ],
        [
            [
                [1, 2, 3],
                [4, 5, 6]
            ],
            [
                [1, 2, 3],
                [4, 5, 6]
            ]
        ],
    ]
)
pprint(a.shape)
b = torch.tensor(
    [
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5]
    ]
)
pprint(b.shape)
c = a @ b
pprint(c.shape)
pprint(c[0, 0, :, :])
pprint(c[1, 0])