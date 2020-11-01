REM A script for updating the package
REM run these commands in Anaconda py3.6 terminal
cd ../dist
del "*.gz" "*.whl"
cd ..
python setup.py sdist bdist_wheel
python -m twine upload dist/*
REM enter your pypi credentials. 
pause

REM install the latest verions locally
REM pip install --upgrade npy2bdv