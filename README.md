# Evaluating the Evaluators: Are Current Few-Shot Learning Benchmarks Fit for Purpose?

This is the official repository for the paper "Evaluating the Evaluators: Are Current Few-Shot Learning Benchmarks Fit for Purpose?" by [Luisa B. Shimabucoro](https://scholar.google.com/citations?hl=pt-BR&user=IYVqNJAAAAAJ&view_op=list_works), [Timothy M. Hospedales](https://scholar.google.com/citations?user=nHhtvqkAAAAJ&hl) and [Henry Gouk](https://scholar.google.com/citations?user=i1bzlyAAAAAJ&hl).

The paper was accepted at the DMLR (Data-centric Machine Learning Research) Workshop at ICML'23.

[[paper](https://arxiv.org/abs/2307.02732)][[bibtex](#citing-the-paper)]

<div align=center><img src="imgs/intro_img.png" width = 80% height = 80%/></div>

Here you can find all the code necessary to replicate the experiments presented in the paper, which are implemented in PyTorch. For details, see the paper: **[Evaluating the Evaluators: Are Current Few-Shot Learning Benchmarks Fit for Purpose?](https://arxiv.org/abs/2307.02732)**.

The implementations of the algorithms used are from [LibFewShot](https://github.com/RL-VIG/LibFewShot) and we thank the authors for the code provided.

## Setup

We would first like to note that the training and testing were made using a Linux environment and Python 3.6, so any divergences from these two noted points might require some additional changes by the user. To setup the environment following all the required dependencies for training and evaluation, please one of the two methods below:

**pip** - Clone the repository, create a virtual environment and then use the `requirements.txt` file to download all dependencies:
```shell
python3.6 -m venv evalFSLevaluators
source evalFSLevaluators/bin/activate
pip install -r requirements.txt
```

**conda** - Clone the repository, create a conda environment and install the dependencies present in the `requirements.txt`:
```shell
conda create -n python=3.6 evalFSLevaluators
conda install --force-reinstall -y --name evalFSLevaluators -c conda-forge --file requirements.txt
conda activate evalFSLevaluators
```
## Folder Organization

## Training

## Evaluation

## Citing the paper

If you found the paper/repository useful please consider giving a star :star: and citation :t-rex::

```
@misc{shimabucoro2023evaluating,
      title={Evaluating the Evaluators: Are Current Few-Shot Learning Benchmarks Fit for Purpose?}, 
      author={Luísa Shimabucoro and Timothy Hospedales and Henry Gouk},
      year={2023},
      eprint={2307.02732},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```


