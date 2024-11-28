# IOHinspector

**IOHinspector** is a Python package designed for processing, analyzing, and visualizing benchmark data from iterative optimization heuristics (IOHs). Whether you're working with single-objective or multi-objective optimization problems, IOHinspector provides a collection of tools to gain insights into algorithm performance. 

IOHinspector is a work-in-progress, and as such some features might be incomplete or have their call signatures changed or expanded when new updates release. As part of the IOHprofiler framework, our aim is to achieve feature parity with the [](), to allow for simple plotting and analysis via this python package for larger datasets which are unsupported on the [IOHanalyzer web-version](https://iohanalyzer.liacs.nl/). 

## Features

- **Data Processing**: Effiecient import and process benchmark data. We currently only support the file structure from [IOHexperimenter](https://github.com/IOHprofiler/IOHexperimenter), but this will be expanded in future releases. By utlizing polars and the meta-data split, large sets of data can be handled efficiently.  
- **Analysis**: Perform in-depth analyses of single- and multi-objective optimization results. For the multi-objective scenario, a variety of performance indicators are supported (hypervolume, igd+, R2, epsilon), each with the option to flexibly change reference points/set as required. 
- **Visualization**: Create informative plots to better understand the optimization process. This included standard fixed-budget and fixed-target plots, EAF and ECDF visualization and more. 

## Installation

Install IOHinspector via pip:

```bash
pip install iohinspector
```

## Getting Started

To highlight the usage of IOHinspector, we have created two tutorials in the form of jupyter notebooks:
* [Single Objective Tutorial](examples/SO_Examples.ipynb)
* [Multi Objective Tutorial](examples/MO_Examples.ipynb)

## License

This project is licensed under a standard BSD-3 clause License. See the LICENSE file for details.

## Acknowledgments

This work has been estabilished as a collaboration between:
* Diederick Vermetten 
* Jeroen Rook
* Oliver L. Preuß
* Jacob de Nobel
* Carola Doerr
* Manuel López-Ibañez
* Heike Trautmann
* Thomas Bäck

## Cite us

Citation information coming soon!
