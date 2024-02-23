#!/bin/bash

export FLASK_APP=./Site/app.py
export FLASK_DEBUG=True
export FLASK_RUN_PORT=8000
flask run
