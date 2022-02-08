# build models
```
python download.py 0.1 0.1
python clean_data.py
python pre_process_convert.py
python pca_one_hot.py
```

# build containers
change run id to run id from previos step
```
mlflow models build-docker -m data_models/download-390163ce63204871b3d5a28099516880 -n download --enable-mlserver
mlflow models build-docker -m pre_models/clean-e7fd64a377594f8eb23ee691d3637354 -n clean-data --enable-mlserver
mlflow models build-docker -m pre_models/convert-9392b22929f3478a9ae044f956357800i -n convert --enable-mlserver
mlflow models build-docker -m pre_models/transform-107aee5674c246c6bcc7b33ec84849ad -n transform --enable-mlserver
```

# run contaniers
```
docker run  -p 1234:8080 --name download download
docker run  -p 1235:8080 --name clean-data clean-data
docker run  -p 1236:8080 --name convert convert
docker run  -p 1237:8080 --name transform transform
```
