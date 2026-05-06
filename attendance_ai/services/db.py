import mysql.connector
from flask import g

def get_connection():
    if "db" not in g:
        g.db = mysql.connector.connect(
            host="localhost",
            user="root",
            password="your password",
            database="db name"
        )
    return g.db


def close_db(e=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_app(app):
    app.teardown_appcontext(close_db)