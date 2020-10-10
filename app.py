import os
import json
from flask import Flask, render_template
import auth, logging


def create_app(test_config=None) -> Flask:
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)

    # use the modified encoder class to handle ObjectId & datetime object while jsonifying the response.
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

    # a simple page that says hello
    @app.route('/')
    def welcome():
        return render_template("welcome.html")

    return app


if __name__ == "__main__":
    app = create_app()
    # This is so that we can configure the host and port from the config .settings.cfg
    # and simply run `python app.py`
    # There is no way to run using `flask run` and have flask use a host or port from
    # a config file.
    #
    # YOu can alternatively run `flask run --host <HOSTNAME> --port <PORT>`
    app.run(
        host=app.config.get("FLASK_SERVER_HOST"),
        port=app.config.get("FLASK_SERVER_PORT"),
    )
