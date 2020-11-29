from flask_wtf import FlaskForm
from wtforms import Form
from wtforms import StringField, PasswordField, BooleanField, SubmitField, SelectField
from wtforms.validators import ValidationError, DataRequired, Email, EqualTo


class RegistrationForm(FlaskForm):
    full_name = StringField("Doctor's Name", validators=[DataRequired()])
    email = StringField("Doctor's Email", validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    password_confirmation = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Register')

# Make a LoginFormm Class with values email and password.
class LoginForm(FlaskForm):
    full_name = StringField("Name", validators=[DataRequired()])
    email = StringField("Doctor's Email", validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')
