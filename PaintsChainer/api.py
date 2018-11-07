'''
File: api.py
Project: PaintsChainer
File Created: Saturday, 27th October 2018 8:32:43 pm
Author: xiaofeng (sxf1052566766@163.com)
-----
Last Modified: Monday, 29th October 2018 12:53:15 pm
Modified By: xiaofeng (sxf1052566766@163.com>)
-----
Copyright 2018.06 - 2018 onion Math, onion Math
'''

import logging
import logging.config
import os
import random
import time
import requests
import yaml
import sys
from flask_wtf import Form
from wtforms import *
from wtforms.validators import Required
from flask import (Flask, redirect, render_template, request,
                   send_from_directory, url_for, jsonify)
from werkzeug import secure_filename
from cgi import parse_header, parse_multipart
from urllib.parse import parse_qs


# sys.path.append('./cgi-bin/wnet')
sys.path.append('./cgi-bin/paint_x2_unet')
import cgi_exe
sys.path.append('./cgi-bin/helpers')


def setup_logging(
    default_path='./log/logging.yaml',
    default_level=logging.INFO,
    env_key='my_logging.yaml '
):
    """Setup logging configuration

    """
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = yaml.load(f.read())
        logging.config.dictConfig(config)
    else:
        lo


logging.config.fileConfig('./log/logging.conf')

logging.debug('debug message')
logging.info("info message")
logging.warn('warn message')
logging.error("error message")
logging.critical('critical message')
logging.log(10, 'log')

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'Flask/upload'
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024

Basedir = os.path.abspath(os.path.dirname(__file__))
upload_dir = os.path.join(Basedir, app.config['UPLOAD_FOLDER'])
if not os.path.exists(upload_dir):
    os.makedirs(upload_dir)

painter = cgi_exe.Painter(gpu=-1)


def create_uuid():
    logger = logging.getLogger(__name__)
    logger.info('create unique name for the uploaded img...')
    nowTime = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
    nowTime_cpu = str(int(round(time.time() * 1000)))
    randomNum = random.randint(0, 10000)
    if randomNum <= 1000:
        randomNum = '{:05d}'.format(randomNum)
    uniqueName = nowTime + '_' + nowTime_cpu + '_' + str(randomNum)
    return uniqueName


def allow_image_file(filename):
    logger = logging.getLogger(__name__)
    logger.info('Distinguish the upload img format')
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)


@app.route('/')
def home():
    logger = logging.getLogger(__name__)
    logging.info('Now in the index page')
    return render_template('base.html')


@app.route('/index')
def index():
    logger = logging.getLogger(__name__)
    logging.info('Now in the index page')
    return render_template('index.html')


@app.route('/index', methods=['POST', 'GET'], strict_slashes=False)
def upload_file():
    # form = MockCreate()
    # print('submit_img:', form['submit_img'])
    # print('submit_color:', form['submit_color'])

    print('request:', request.args, request.method, request.files, request._get_current_object())
    if request.method == 'POST' and 'upload_img' in request.files:
        img = request.files['upload_img']
        if not (img and allow_image_file(img.filename)):
            return jsonify({"error": 1001, "msg": "请检查上传的图片类型，仅限于png、PNG、jpg、JPG、bmp"})
        upload_fname = secure_filename(img.filename)
        new_fname = create_uuid() + '_' + upload_fname
        img.save(os.path.join(upload_dir, new_fname))
        img_url = url_for('uploaded_file', filename=new_fname)
        print('img_url:', img_url)
        return render_template('index.html', img=img_url)
    # else:
    #     return render_template('base.html')
        # if os.path.exists(img_url):
        #     painter.colorize()
    return render_template('index.html')


@app.route('/howto')
def howto():
    render_template('howto.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=1989, debug=True)
