# medusa-search
===============

[![DOI](https://zenodo.org/badge/898711582.svg)](https://doi.org/10.5281/zenodo.14279139)

The repository contains code for machine learning (ML)-powered search engine specifically tailored for analyzing tera-scale high-resolution mass spectrometry (HRMS) data.
 
[Follow us](http://ananikovlab.ru)

## How to use it?

1) First, you have to install [MEDUSA python package](https://github.com/Ananikov-Lab/medusa).

2) In addition, you should add folder with MEDUSA package to PYTHON PATH to make importing functions available in running scripts and notebooks without changing it. Make the below command in the root directory of MEDUSA.

```bash
export PYTHONPATH := $(shell pwd):$(PYTHONPATH)
```

3) After that, you have to install additional requirements in medusa_search repository. It is reccomended to use the same virtual environment for MEDUSA and medusa_search.

```bash
pip install -r requirements.txt
```

4)Also, specify path to MEDUSA repository in **config.yaml** file.

4) Moreover, you have to specify path to database of HRMS spectra in **config.yaml** or do it manually in **create_batches.sh** and **create_unique_batch.sh**.

5) The description of how to use MEDUSA Search CLI can be found in the Supplemetary Information of the article.
