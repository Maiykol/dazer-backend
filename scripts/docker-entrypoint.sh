#!/bin/bash

python3 manage.py makemigrations
python3 manage.py migrate
# python3 manage.py createfixtures
# python3 manage.py cleanuptasks

/usr/bin/supervisord -c "/etc/supervisor/conf.d/supervisord.conf"