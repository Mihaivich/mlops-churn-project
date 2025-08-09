import flask, joblib, pandas as pd, os
model = joblib.load(os.path.join("/app/model", "model.joblib"))
app = flask.Flask(__name__)

@app.route("/ping", methods=["GET"])
def ping():
    return flask.Response(response="\n", status=200, mimetype="application/json")

@app.route("/invocations", methods=["POST"])
def invocations():
    data = flask.request.get_json(force=True)
    df = pd.DataFrame(data['instances'])
    try:
        predictions = model.predict_proba(df)[:, 1]
        return flask.Response(response=flask.json.dumps({'predictions': predictions.tolist()}), status=200, mimetype="application/json")
    except Exception as e:
        return flask.Response(response=str(e), status=500, mimetype="text/plain")