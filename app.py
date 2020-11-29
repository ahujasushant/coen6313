import os
import flask

from flask import Flask, render_template
import bcrypt
from pymongo import MongoClient

import auth
import logging
from forms import *


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
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    app.logger.setLevel(logging.DEBUG)
    app.register_blueprint(auth.bp)

    client = MongoClient('localhost', 27017)
    db = client['COEN-6313']
    hospital_collection = db['hospitals']
    doctor_collection = db['doctors']
    image_collection = db['images']

    @app.route('/')
    def welcome():
        return render_template("welcome.html")

    @app.route('/register', methods=['GET', 'POST'])
    def register():
        form = RegistrationForm()
        if form.validate_on_submit():
            existing_user = doctor_collection.find({'name': form.full_name.data})
            if existing_user is None:
                hashpass = bcrypt.hashpw(form.password.data.encode('utf-8'), bcrypt.gensalt())
                doctor = {"full_name": form.full_name.data, "email": form.email.data, "password": hashpass}
                doctor_collection.insert_one(doctor)
                flask.session['full_name'] = form.full_name.data
                return flask.redirect(flask.url_for('welcome'))
            return 'That email already exists!'

        return render_template('register.html',form=form)

    # Create a method for login and control it's session values
    @app.route('/login', methods=['GET', 'POST'])
    def login():
        form = LoginForm()
        if form.validate_on_submit():
            login_user = doctor_collection.find_one({'full_name': form.full_name.data})
            if login_user:
                if bcrypt.checkpw(form.password.data.encode('utf-8'),login_user['password']):
                    flask.session['full_name'] = form.full_name.data
                    return flask.redirect(flask.url_for('welcome'))
            return 'Invalid username/password combination'

        return flask.render_template('login.html',form=form)

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(
        host=app.config.get("FLASK_SERVER_HOST"),
        port=app.config.get("FLASK_SERVER_PORT"),
    )
