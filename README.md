# How to build and run

## Build the image

Install docker
cd in directory containing the project's Dockerfile, then:

```
docker build -t think
```

The first time, the build can take several minutes. Once it is completed:

## Run the container

```
docker run -p 8888:8888 think
```

The container will print an url with token to access Jupyter Notebook.

NOTE: by default, data is not persistent.

## Start a running container

List the running containers:
```
docker ps -a
```

Start container:
```
docker start -i container_name_or_id
```
