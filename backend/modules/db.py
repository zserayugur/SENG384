
import pymysql

def get_db_connection():
    connection = pymysql.connect(
        host="127.0.0.1",
        user="root",
        password="",  # XAMPP genelde boş
        database="facial_app",
        port=3307,
        cursorclass=pymysql.cursors.DictCursor
    )
    return connection