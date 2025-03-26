# napari-vos-sam2

<!---[![License MIT](https://img.shields.io/pypi/l/napari-vos-sam2.svg?color=green)](https://github.com/ledvic/napari-vos-sam2/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-vos-sam2.svg?color=green)](https://pypi.org/project/napari-vos-sam2)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-vos-sam2.svg?color=green)](https://python.org)
[![tests](https://github.com/ledvic/napari-vos-sam2/workflows/tests/badge.svg)](https://github.com/ledvic/napari-vos-sam2/actions)
[![codecov](https://codecov.io/gh/ledvic/napari-vos-sam2/branch/main/graph/badge.svg)](https://codecov.io/gh/ledvic/napari-vos-sam2)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-vos-sam2)](https://napari-hub.org/plugins/napari-vos-sam2)
-->
# Video Object Segmentation using pre-trained SAM2 model

This [napari] plugin was generated with [copier] using the [napari-plugin-template].

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/napari-plugin-template#getting-started

and review the napari docs for plugin developers:
https://napari.org/stable/plugins/index.html
-->

<!-- <video src="assets/demo.mp4" controls width="640">
</video> -->

<video src="https://raw.githubusercontent.com/ledvic/napari-vos-sam2/main/assets/assets/demo.mp4" controls width="640">
</video>
----------------------------------

## Project Title

    #IDP32| IBDM | Automated tracking and segmentation of cell aggregates aspiration experiments 

## Installation

(Recommend) Create a conda environment:
    
    conda create -n napari-vos-env python=3.11
    conda activate napari-vos-env
    
Install PyTorch with GPU 

    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    
Install napari

    python -m pip install "napari[all]"

Install SAM 2 with a stable version before 30 September 2024 

    git clone https://github.com/facebookresearch/sam2.git
    cd sam2
    git rev-list -n 1 --before="2024-09-30" HEAD

    git checkout <commit_hash>
    pip install -e.

Install the plugin

    pip install git+https://github.com/ledvic/napari-vos-sam2.git


To update latest development version :

    conda activate napari-vos-env
    pip install -U git+https://github.com/ledvic/napari-vos-sam2.git


Export mask as TIF

- Convert the data type of selected label layer int a signed 16-bit integer format 
by right clicking on it and selecting: *Convert data type > Convert to int16*
- Save the selected layer as TIFF file from the menu: *File > Save Selected Layers..*
- This TIFF file can be opened in ImageJ/Fiji

    


To run the plugin properly when re-opening it, make sure the current folder is the SAM 2 folder (i.e., **sam2** from above installation)


## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [MIT] license,
"napari-vos-sam2" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[copier]: https://copier.readthedocs.io/en/stable/
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[napari-plugin-template]: https://github.com/napari/napari-plugin-template

[file an issue]: https://github.com/ledvic/napari-vos-sam2/issues

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
