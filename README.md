# medusa-search

[![DOI](https://zenodo.org/badge/898711582.svg)](https://doi.org/10.5281/zenodo.14279139)

The repository contains code for machine learning (ML)-powered search engine specifically tailored for analyzing tera-scale high-resolution mass spectrometry (HRMS) data.
 
[Follow us](http://ananikovlab.ru)

## How to use it?

1) First, you have to install [MEDUSA python package](https://github.com/Ananikov-Lab/medusa).

2) After that, you have to install additional requirements in medusa-search repository. It is reccomended to use the same virtual environment for medusa and medusa-search.

```bash
pip install -r requirements.txt
```

3) Change file permissions for bash-scripts inside *search* folder.

```bash
chmod 744 search/*.sh
```

4) Now you can run *setup.py* file and follow the instructions. You will need to specify the path to the MEDUSA package folder and to HRMS database. 

```bash
python setup.py
```

5) Every time you perform search procedure, **medusa-search** takes functions from *medusa_repository_path* folder. You can create a .pth file in the site directory to add *medusa_repository_path*.

Site directory can be found with the command below:

```bash
python -m site --user-site
```

```bash
# find directory
SITEDIR=$(python -m site --user-site)

# create new .pth file with medusa_repository_path
echo "<your medusa repository path>" > "$SITEDIR/medusa.pth"
```

6) To use command-line interface you should always go to *search* folder and run *main.py* script

```bash
cd search
```

```bash
python main.py
```

7) the procedure of batches creation and indexing should be performed before search. It can be performed with **create_batches** and **index** commands respectively. 

8) After indexing, **search** can be performed.

P.S. More explanations can be found in the Supporting information of the article.
