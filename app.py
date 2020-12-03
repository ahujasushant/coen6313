import os
from typing import Dict

import flask
import numpy as np
from flask import Flask, render_template, request
import bcrypt
import pymongo
import service
import dns
from service import HeatMapGenerator, DISEASES

import auth
import logging
from forms import *

THRESHOLDS = [0.1] * 14


def create_app(test_config=None) -> Flask:
    app = Flask(__name__, instance_relative_config=True)

    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite'),
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        app.config.from_mapping(test_config)

    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    app.logger.setLevel(logging.DEBUG)
    app.register_blueprint(auth.bp)

    client = pymongo.MongoClient(
        "mongodb+srv://coen-6313:UetDz4VzcAmvyM0w@coen-6313.8dgrr.mongodb.net/coen_6313?retryWrites=true&w=majority")
    db = client.coen_6313
    hospital_collection = db.hospitals
    doctor_collection = db.doctors
    image_collection = db.images

    @app.route('/')
    def welcome():
        return render_template("welcome.html")

    @app.route('/register', methods=['GET', 'POST'])
    def register():
        form = RegistrationForm()
        if form.validate_on_submit():
            existing_user = doctor_collection.find_one({'email': form.email.data})
            if existing_user is None:
                hashpass = bcrypt.hashpw(form.password.data.encode('utf-8'), bcrypt.gensalt())
                doctor = {"full_name": form.full_name.data, "email": form.email.data, "password": hashpass}
                doctor_collection.insert_one(doctor)
                flask.session['full_name'] = form.email.data
                return flask.redirect(flask.url_for('welcome'))

        return render_template('register.html', form=form)

    # Create a method for login and control it's session values
    @app.route('/login', methods=['GET', 'POST'])
    def login():
        form = LoginForm()
        if form.validate_on_submit():
            login_user = doctor_collection.find_one({'email': form.email.data})
            if login_user:
                if bcrypt.checkpw(form.password.data.encode('utf-8'), login_user['password']):
                    flask.session['full_name'] = form.email.data
                    return flask.redirect(flask.url_for('welcome'))

        return flask.render_template('login.html', form=form)

    def diagnose_img(img_path: str) -> Dict:
        f_name = img_path
        predictions_tensor = service.diagnose(f_name)
        threshold_result = []
        for r in predictions_tensor:
            threshold_result.append(r.numpy() > np.array(THRESHOLDS))
        threshold_result = threshold_result[0].tolist()
        result = {}
        heat_map = HeatMapGenerator()
        heat_map.generator(f_name, './static/heat_maps/' + f_name)
        heat_map_image = '/heat_maps/' + f_name
        result["heat_map_image"] = heat_map_image

        predictions = predictions_tensor.squeeze().tolist()
        result["predictions"] = predictions
        result["threshold_result"] = threshold_result
        if True in threshold_result:
            import requests

            URL = "https://discover.search.hereapi.com/v1/discover"
            latitude = 45.5017
            longitude = -73.5673
            api_key = '2SIwAzBiMjzkjmpBa3rqv2cETbWiPbOaedzsbDmsSQI'
            query = 'hospitals'
            limit = 5

            PARAMS = {
                'apikey': api_key,
                'q': query,
                'limit': limit,
                'at': '{},{}'.format(latitude, longitude)
            }

            # sending get request and saving the response as response object
            r = requests.get(url=URL, params=PARAMS)
            data = r.json()
            result["recommendation"] = data
        result["diseases"] = DISEASES
        return result

    @app.route('/rest/diagnose_image', methods=['POST'])
    def diagnose_image_rest():
        form = ImageForm()
        f_name = form.image.data.filename
        form.image.data.save(f_name)
        result = diagnose_img(f_name)
        os.remove(f_name)
        return result

    @app.route('/diagnose_image', methods=['GET', 'POST'])
    def diagnose_image():
        form = ImageForm()
        if request.method == 'POST':
            f_name = form.image.data.filename
            form.image.data.save(f_name)
            result = diagnose_img(f_name)
            os.remove(f_name)
            return flask.render_template('image_results.html', count=5, **result)

        return flask.render_template('image_form.html', form=form)

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(
        host='0.0.0.0',
        port=8000,
    )
