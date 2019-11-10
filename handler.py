from test import prediction
from flask import Flask,request,render_template
app = Flask(__name__)

@app.route('/')
def root():
	return render_template("file_upload.html",title="CaptionBot")
@app.route('/run',methods=['GET'])
def start_prediction():
	description = prediction()
	return description

if __name__ == '__main__':
	app.run(debug=True,port=8081)


