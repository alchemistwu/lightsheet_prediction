# lightsheet
## 1. Depoyment version: [https://drive.google.com/file/d/1RQY8bhExJU8tp7iX4hlzvd-ENUsCZ2uf/view?usp=sharing]
## 2. To use the spec file to create an identical environment on the same machine or another machine:
* `conda create --name myenv --file spec-file.txt`
## 3. To use the spec file to install its listed packages into an existing environment:
* `conda install --name myenv --file spec-file.txt`
## 4. To create your own version of .exe file: use `pyinstaller lspredict.spec` 
* Find the .exe file under `src/dist/lsPredict`
* Also copy the weight folder to `src/dist`
