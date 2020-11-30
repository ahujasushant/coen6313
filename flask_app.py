from flask import Flask, render_template
from flask_bootstrap import Bootstrap

# create the object of Flask
app = Flask(__name__)

bootstrap = Bootstrap(app)


# creating our routes
@app.route('/')
def index():
    return render_template('index.html', content="data")


# contact routes
@app.route('/contact')
def contact():
    return render_template('contact_doctors.html')


# run flask app
if __name__ == "__main__":
    app.run(debug=True)
