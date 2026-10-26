REM A script for updating the package on PyPI
REM run in the npy2bdv mamba env: mamba activate npy2bdv
cd ./dist
del "*.gz" "*.whl"
cd ..
python -m build
python -m twine upload dist/*
pause

REM install the latest version locally
REM pip install --upgrade npy2bdv
