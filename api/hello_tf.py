from flask import Flask
import tensorflow as tf

app = Flask(__name__)

hello = tf.constant("Hello Tensorflow")
"""
g = tf.Graph()
with g.as_default():
    hello = tf.constant("Hello Tensorflow")
"""
@app.route("/")
def hello_ts():
    sess = tf.Session()#graph = g)
    foo = str(sess.run())
    return foo

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8000)
