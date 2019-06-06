# Web Development

This directory have multiple apps that can created on a docker image using shell scripts in `tasks/` alongside with a few new files here in `api/`. The docker images then can be loaded. 

TODO: shove `hello_world_api` onto AWS Lambda
TODO: create some test codes to test the connections in case something goes wrong in our deployed service.

## 1. Hello World!

Test hello_world.py locally:
```
pipenv run python api/hello_world.py
```
The webpage `http://0.0.0.0:5000/` should read `Hello World!`

We can then create the docker image using
```
tasks/build_hello_world_api_docker.sh
```

Note in the .sh file we are using `requirements.txt` for the whole project so its heavier in weight.

It will take a few mins to build.

After its done, we can run the docker image:
```
docker run -p 5000:5000 --name api -it --rm hello_world_api
```
I think 5000:5000 is the port it listens to but no idea what other paramenters mean. Yet.

The webpage can be loaded the same way as above.

To see the list of docker containers running, do
```
docker container ls -a
```
To see a list of docker images, do
```
docker images -a
```

Need to install the serverless framework.
```
cd api
npm install
```
modify `serverless.yml`

And run 
```
sls config credentials --provider aws --key AKIAIOSFODNN7EXAMPLE --secret wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
```
if sls not found, do
```
npm config set prefix /usr/local

sudo npm i -g serverless
```
and install

```
sudo sls plugin install -n serverless-python-requirements
```

and this module to optimize sls 
```
npm install serverless-optimizer-plugin --save
```

my local npm read/write access is fucked up so gotta run sudo in front of npm commands

