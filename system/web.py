from flask import  Flask, render_template, request, redirect, url_for
from datetime import datetime
import mysql.connector
app = Flask(__name__,static_folder="static",template_folder="templates")


mydb = mysql.connector.connect(
    host="localhost",
    user="yrq",
    password="yrq",
    database="ship_web"
)


@app.route("/register",methods=["GET"])
def register_index():
    return render_template("register.html")


@app.route("/register",methods=["POST"])
def register():
    message = None
    username = request.form["reg-username"]
    password = request.form["reg-password"]
    print(username, password)

    mycursor = mydb.cursor()
    sql = "SELECT * FROM users WHERE username = %s"
    val = (username,)
    mycursor.execute(sql, val)
    existing_user = mycursor.fetchone()
    if existing_user:
        message ="error"
    else:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sql = "INSERT INTO users (username, password, created_at) VALUES (%s, %s, %s)"
        val = (username, password, now)
        mycursor.execute(sql, val)
        message ="success"
    mydb.commit()
    mycursor.close()
    
    return render_template("register.html", message=message)



@app.route("/login",methods=["POST"])
def login():
    message = None
    username = request.form["username"]
    password = request.form["password"]
    print(username, password)

    mycursor = mydb.cursor()
    sql = "SELECT * FROM users WHERE username = %s AND password = %s"
    val = (username,password,)
    mycursor.execute(sql, val)
    existing_user = mycursor.fetchone()
    if existing_user:
        message ="success"
    else:
        message ="nouser"
    mydb.commit()
    mycursor.close()
    
    if message=="success":
         return redirect("http://121.43.138.58:7860/")
    else:
        return render_template("register.html", message=message)
    

@app.route("/")
@app.route("/index")
@app.route("/home")
def index():
    return render_template("login.html")

if __name__ == "__main__":
    app.run(debug=False)
