#!/bin/sh
echo Running Scripts
pip install -r requirements.txt
jupyter nbextension enable --py --sys-prefix widgetsnbextension
wget https://www.dropbox.com/s/yxg7nuhrkiwiz2b/data.zip
unzip data.zip
echo "It's all set"
