# Code Summary #
`aggregate.py`: this script aggregates the scores outputted by `expriments.py`

`experiments.py`: This script runs UNAVIODS and some comparison algorithms across a grid of parameters on a window specified by a command line argument.

`genPrevelanceArrs.py`: Thus script creates `Pis_\[i\].npy` for each i, where i corresponds to the index of the i^th prevelance in the array Pis each `Pis_\[i\].npy` contains a `n' < n` length vector of indicies, such that the prevelance of outliers in the subset pointed to by these indicies corresponds to the ith value of Pis.

`genSimulatedDataSet.py`: This script generates a simulated dataset

`genSimulationNCDFs.py`: This script finds the NCDFs for simulated data using different norms and numbers of features, then creates CSV files for LaTeX to plot the NCDFs

`loadAndCleanDoH.py`: This file loads and cleans the data from the `CIRA-CIC-DoHBrw-2020` data set

`loadAndCleanIDS.py`: This file loads and cleans the data from the `CICIDS2017` data set

`txtToLatex.py`: This file loads the TXT file outputted by `aggregate.py` and creates several CSV's saved in `\[outfolder\]`. These can be used by our LaTeX code to create latex plots.

`txtToLatex.py`: the unavoids outlier detection and visualization library. All functions have a docstring describe their use.

`main.py`: basic usuage example of the unavioids library

# References #
W. A. Yousef, I. Traoré and W. Briguglio, "UN-AVOIDS: Unsupervised and Nonparametric Approach for Visualizing Outliers and Invariant Detection Scoring," in IEEE Transactions on Information Forensics and Security, vol. 16, pp. 5195-5210, 2021, doi: 10.1109/TIFS.2021.3125608.

# Citation #
Please, cite this work as:

```
@ARTICLE{Yousef2021UNAVOIDS,
  author={Yousef, Waleed A. and Traoré, Issa and Briguglio, William},
  journal={IEEE Transactions on Information Forensics and Security}, 
  title={UN-AVOIDS: Unsupervised and Nonparametric Approach for Visualizing Outliers and Invariant Detection Scoring}, 
  year={2021},
  volume={16},
  number={},
  pages={5195-5210},
  doi={10.1109/TIFS.2021.3125608}}
```
