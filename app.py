import os
import flask

from flask import Flask, render_template
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
            # Filter the details from the input form
            # Store them into doctors collection
            flask.flash('Congrats')
            return flask.redirect(flask.url_for('welcome'))
        return render_template("register.html", form=form)

    # Create a method for login and control it's session values

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(
        host=app.config.get("FLASK_SERVER_HOST"),
        port=app.config.get("FLASK_SERVER_PORT"),
    )
