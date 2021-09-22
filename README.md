# Unsupervised Abstract Reasoning for Raven’s Problem Matrices

This code is the implementation of our TIP [paper](https://arxiv.org/pdf/2109.10011.pdf)

This is the first unsupervised abstract reasoning method on Raven's Progressive Matrices, it is an extention of our arxiv preprint[paper](https://arxiv.org/pdf/2002.01646.pdf)

## Comparision with some supervised methods. 

### Average testing accuracy on the RAVEN, I-RAVEN, and PGM dataset

|      Method       |   Raven  |    I-RAVEN     |     PGM       |
|-------------------|----------|----------------|---------------|
| CNN               |  36.97   |     13.26      |     33.00     |
| ResNet50          |  86.26   |        -       |     42.00     |
| DCNet (ICLR2021)  |**93.58** |   **49.36**    |     68.57     |
|    NCD (Ours)     |  36.99   |     48.22      |     47.62     |



### Generalization test results on PGM dataset

|      Method       | neutral| interpolation  | extrapolation |
|-------------------|--------|----------------|---------------|
| WReN (ICML2018)   |  62.6  |     64.4       |     17.2      |
| DCNet (ICLR2021)  |  68.6  |     59.7       |     17.8      |
| MXGNet (ICLR2020) |**89.6** | **70.85**     |     18.4      |
|    NCD (Ours)     | 47.6   |     47.0       |   **24.9**    |


## Citation
If our code is useful for your research, please cite the following papers.

```
@article{zhuo2021unsup,
  title={Unsupervised Abstract Reasoning for Raven’s Problem Matrices},
  author={Tao Zhuo, Qiang Huang, and Mohan Kankanhalli},
  journal={IEEE Transactions on Image Processing},
  year={2021}
}
```

```
@article{zhuo2020solving,
  title={Solving Raven's Progressive Matrices with Neural Networks},
  author={Tao Zhuo and Mohan Kankanhalli},
  journal={arXiv preprint arXiv:2002.01646},
  year={2020}
}
```

```
@inproceedings{iclr2021,  
    author={Tao Zhuo and Mohan Kankanhalli},  
    title={Effective Abstract Reasoning with Dual-Contrast Network},  
    booktitle={International Conference on Learning Representations (ICLR)},      
    year={2021}
}
```
