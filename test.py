====================================================================================================
Layer (type:depth-idx)                             Output Shape              Param #
====================================================================================================
Network                                            [64, 12, 170, 1]          --
├─MGSTGNN: 1-1                                     [64, 12, 170, 1]          --
│    └─Encoder: 2-1                                [170, 12]                 11,256
│    │    └─Embedding: 3-1                         [64, 12, 170, 12]         3,456
│    │    └─Embedding: 3-2                         [64, 12, 170, 12]         84
│    └─DSTRNN: 2-2                                 --                        --
│    │    └─ModuleList: 3-7                        --                        (recursive)
│    │    └─ModuleList: 3-8                        --                        --
│    │    └─ModuleList: 3-9                        --                        (recursive)
│    │    └─ModuleList: 3-6                        --                        1,548
│    │    └─ModuleList: 3-7                        --                        (recursive)
│    │    └─ModuleList: 3-8                        --                        --
│    │    └─ModuleList: 3-9                        --                        (recursive)
====================================================================================================
Total params: 1,226,928
Trainable params: 1,226,928
Non-trainable params: 0
Total mult-adds (M): 50.83
====================================================================================================
Input size (MB): 1.57
Forward/backward pass size (MB): 78.34
Params size (MB): 0.03
Estimated Total Size (MB): 79.94
====================================================================================================

Process finished with exit code 0
