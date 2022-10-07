from flask import Flask, request, render_template
from generator import generate_horoscope, get_input

app = Flask(__name__)

# global: sign names and symbols
'''
signNames = ["Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
         "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"]
symbols = {"Aries": "&#9800;", "Taurus": "&#9801;", "Gemini": "&#9802;",
            "Cancer": "&#9803;", "Leo": "&#9804;", "Virgo": "&#9805;",
            "Libra": "&#9806;", "Scorpio": "&#9807;", "Sagittarius": "&#9808;",
            "Capricorn": "&#9809;", "Aquarius": "&#9810;", "Pisces": "&#9811;"}
'''

# Selects the page for which a function is to be defined. Right now there will only be one page in your website.

@app.route('/')

def index():
    return render_template('index.html')

@app.route("/horoscope", methods=['POST'])
def getSign():
    sign = request.form["sign"]
    input = get_input(sign)
    text = generate_horoscope(sign, input, 70)
    return render_template('horoscope.html', sign=sign, text=text)

# The above function returns the HTML code to be displayed on the page

if __name__ == '__main__':

   app.run(host='0.0.0.0', port=5001, debug=True)
