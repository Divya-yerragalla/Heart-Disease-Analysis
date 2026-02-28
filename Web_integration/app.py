from flask import Flask, render_template

app = Flask(__name__)

# Home Page
@app.route('/')
def home():
    return render_template('index.html')


# Dashboard Page
@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')


# Stories Page
@app.route('/story')
def stories():
    return render_template('story.html')


# About Us Page
@app.route('/about')
def about():
    return render_template('about.html')


# Contact Us Page
@app.route('/contact')
def contact():
    return render_template('contact.html')


if __name__ == '__main__':
    app.run(debug=True)