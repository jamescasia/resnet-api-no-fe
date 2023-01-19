source env/bin/activate
export FLASK_APP="app.main:create_app"
export FLASK_RUN_PORT=8345
echo $VIRTUAL_ENV
flask run