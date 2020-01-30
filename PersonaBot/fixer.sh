#!/bin/bash
echo "starting to install required modules"
source /opt/python/run/venv/bin/activate
echo "env activated"
sudo /opt/python/run/venv/bin/pip --no-cache-dir install --upgrade pip==19.3.1
sudo /opt/python/run/venv/bin/pip --no-cache-dir install flask
sudo /opt/python/run/venv/bin/pip --no-cache-dir install flask-restful
sudo /opt/python/run/venv/bin/pip --no-cache-dir install jsonschema
sudo /opt/python/run/venv/bin/pip --no-cache-dir install torch
sudo /opt/python/run/venv/bin/pip --no-cache-dir install pytorch-ignite
sudo /opt/python/run/venv/bin/pip --no-cache-dir install pytorch-transformers==1.2
sudo /opt/python/run/venv/bin/pip --no-cache-dir install tensorboardX==1.8
sudo /opt/python/run/venv/bin/pip --no-cache-dir install tensorflow==1.5
deactivate
echo "env de-activated"
sudo mkdir /home/wsgi
sudo chmod 764 /home/wsgi
echo "done !"