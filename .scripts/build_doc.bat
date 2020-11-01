REM build the docs using pdoc, run these in Anaconda prompt
pdoc npy2bdv --html -o ./docs
move /y .\docs\npy2bdv\*.html .\docs
rmdir /Q .\docs\npy2bdv
pause