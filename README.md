# medusa-search

[![DOI](https://zenodo.org/badge/898711582.svg)](https://doi.org/10.5281/zenodo.14279139)

The repository contains code for machine learning (ML)-powered search engine specifically tailored for analyzing tera-scale high-resolution mass spectrometry (HRMS) data.
 
[Follow us](http://ananikovlab.ru)

## How to use it?

1) First, you have to git clone and install dependencies from [MEDUSA python package](https://github.com/Ananikov-Lab/medusa). It is recommended to use the same Conda environment with Python 3.8 for medusa and medusa search. Then you have to change branch to `medusa_search`.

```bash
conda create -n medusa python=3.9 pip=24.0 setuptools=65.5.0
git checkout medusa_search
pip install -r requirements.txt
```

2) After that, you have to clone `medusa-search` repository and install additional requirements.

```bash
pip install -r requirements.txt
```

3) Change file permissions for bash-scripts inside `search` folder.

```bash
chmod 744 search/*.sh
```

4) Now you can run `setup.py` file and follow the instructions. You will need to specify the path to the MEDUSA package folder and to HRMS database. 

```bash
python setup.py
```

5) Every time you perform search procedure, medusa-search takes functions from *medusa_repository_path* folder. You can create a .pth file in the site directory to add *medusa_repository_path*.

```bash
# find site directory
SITEDIR=$(python -c "import site; print(site.getsitepackages()[0])")

# create if site directory doesn't exist
mkdir -p "$SITEDIR"

# create new .pth file with medusa_repository_path
echo "<your medusa repository path>" > "$SITEDIR/medusa.pth"
```

6) To use command-line interface you should always go to `search` folder and run *main.py* script

```bash
cd search

python main.py
```

7) the procedure of batches creation and indexing should be performed before search. It can be performed with `create_batches` and `index` commands respectively. 

8) After indexing, `search` can be performed. Results are saved in `medusa-search/search/reports` folder.

**Most important commands:**
| Command | Description |
| --- | --- |
| `create_batches` | Create batches (or shards) of spectra filenames |
| `create_unique_batch` | Create batch only with spectra filenames that have specific word indicator |
| `index` | Index filenames from batches located in `medusa-search/search/batches` directory and save results in `medusa-search/search/index_pickles` directory |
| `search` | Search formula in spectra indexed in specific directory. Results are saved in `medusa-search/search/reports`|

P.S. More explanations can be found in the Supporting information of the article.
