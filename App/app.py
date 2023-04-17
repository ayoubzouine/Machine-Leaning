from flask import Flask  
from views import views

app = Flask(__name__)
app.secret_key = 'my-secret-key'

app.register_blueprint(views,url_prefix="/Skin-Disease-Predictor") 


if __name__=='__main__':
    app.run(debug=True,port=9000)