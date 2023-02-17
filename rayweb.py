from flask import *
from raytensor import RayTensor

upload_file = "static/images/test.jpg"
app = Flask(__name__)
raytensor = RayTensor()
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1] in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/xray-request", methods=["post", "get"])
def xray_scan():
    if request.method == "POST":
        image = request.files["image"]
        if image and allowed_file(image.filename):
            image.save(upload_file)
            return redirect("/xray-result")
        else:
            return redirect("/invalid-format")

    return render_template("xray-request.html")


@app.route("/xray-result")
def xray_result():
    predict = raytensor.xray_predict(upload_file)
    return render_template("xray-result.html", predict=predict)


@app.route("/ct-request", methods=["post", "get"])
def ct_scan():
    if request.method == "POST":
        image = request.files["image"]
        if image and allowed_file(image.filename):
            image.save(upload_file)
            return redirect("/ct-result")
        else:
            return redirect("/invalid-format")

    return render_template("ct-request.html")


@app.route("/ct-result")
def ct_result():
    predict = raytensor.ct_predict(upload_file)
    return render_template("сt-result.html", predict=predict)


@app.route("/in-development")
def dev():
    return render_template("in_development.html")


@app.route("/invalid-format")
def invalid_format():
    return render_template("error-invalidformat.html")


@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0")
