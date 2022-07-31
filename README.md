# Ranking Loss: Maximizing the Success Rate in Deep Learning Side-Channel Analysis
The current repository is associated with the article "<a href='https://tches.iacr.org/index.php/TCHES/article/view/8726'>Ranking Loss: Maximizing the Success Rate in Deep Learning Side-Channel Analysis</a>" available on <a href='https://tches.iacr.org/index.php/TCHES/index'>IACR Transactions on Cryptographic Hardware and Embedded Systems (TCHES)</a> and the <a href='https://eprint.iacr.org/2020/872'>eprints</a>.

The optimized Ranking Loss has been developed by Pritha Gupta (credit: <a href='https://github.com/gabzai/Ranking-Loss-SCA/issues/1'>issue 1</a>) and validated on Tensorflow 2.0.

Each dataset is composed of the following scripts and repositories:
- <b>cnn_architecture.py</b>: provides the script in order to train the model introduced in the article,
- <b>exploit_pred.py</b>: computes the evolution of the right key and saves the resulted picture (<b>Credit</b>: Damien Robissout),
- <b>(Optionnal) clr.py</b>: computes the One-Cycle Policy (see "<a href='https://arxiv.org/abs/1708.07120'>Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates
</a>" and "<a href='https://arxiv.org/abs/1803.09820'>A disciplined approach to neural network hyper-parameters: Part 1 -- learning rate, batch size, momentum, and weight decay</a>,
- <b>"training_history"</b>: contains information related to the loss function, the accuracy
- <b>"model_predictions"</b>: contains information related to the model predictions,
- <b>"fig"</b>: contains the figure related to the rank evolution.
- <b>"..._trained_models"</b>: containts the model used in the article.

The trace sets were obtained from publicly databases: 
- <b>AES_HD dataset</b>: https://github.com/AESHD/AES_HD_Dataset
- <b>ASCAD</b>: https://github.com/ANSSI-FR/ASCAD

## Raw data files hashes
The zip file SHA-256 hash value is:
<hr>

**AES_HD/AES_HD_dataset.zip:**
`00a3d02f01bae8c4fcefda33e3d1adb57bed0509ded3cdcf586e213b3d87e41b`

<hr>

**ASCAD/Desync0/ASCAD_dataset.zip:**
`5f5924e2d0beca5b57fbc48ace137dbb2fe12dd03976aa38f4a699ab21e966b0`

**ASCAD/Desync50/ASCAD_dataset.zip:**
`9bf704727390a73cf67d3952bc2cacef532b0b62e55f85d615edaa6cd8521f51`

**ASCAD/Desync100/ASCAD_dataset.zip:**
`2d803db27e58fec3d805cd3cf039b303cad1e0c9ea7a8102a07020bd07113cd9`

<hr>

## Citation

If you use our code, models or wish to refer to our results, please use the following BibTex entry:
```
@article{Zaid_Bossuet_Dassance_Habrard_Venelli_2020, 
title={Ranking Loss: Maximizing the Success Rate in Deep Learning Side-Channel Analysis}, 
volume={2021}, 
url={https://tches.iacr.org/index.php/TCHES/article/view/8726}, 
DOI={10.46586/tches.v2021.i1.25-55}, 
number={1}, 
journal={IACR Transactions on Cryptographic Hardware and Embedded Systems}, 
author={Zaid, Gabriel and Bossuet, Lilian and Dassance, François and Habrard, Amaury and Venelli, Alexandre}, 
year={2020}, 
month={Dec.}, 
pages={25-55} 
}
```
