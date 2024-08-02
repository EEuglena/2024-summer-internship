import torch


pos = torch.tensor(
    [
        [
            1,
            2,
            3,
        ],
        [
            4,
            5,
            6,
        ],
        [
            7,
            8,
            9,
        ],
    ]
)

pos1 = pos.unsqueeze(0)
pos2 = pos.unsqueeze(1)
print(pos1)
print(pos2)

rpos = pos1 - pos2
print(rpos)

distance = torch.sum(rpos.square(), 2).sqrt()

print(distance)
