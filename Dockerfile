# pull official base image
FROM python:3.11.6

# user for supervisord
# RUN adduser --system --group --no-create-home appuser

# set work directory
WORKDIR /usr/src/app

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN apt-get update
RUN apt-get install -y supervisor nginx libgtk-3-dev

# install dependencies
COPY ./requirements.txt /usr/src/app/requirements.txt
RUN pip install -r requirements.txt

COPY ./supervisord.conf /etc/supervisor/conf.d/supervisord.conf

COPY . /usr/src/app/

# switch to non-root user
# USER appuser