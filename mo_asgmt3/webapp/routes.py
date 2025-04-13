from webapp import app, db
from flask import render_template, request, redirect, url_for, jsonify
from models import DataPoint
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


# Website methods

@app.route('/')
def home():
    try:
        data_points = db.session.execute(db.select(DataPoint)).scalars().all()
        return render_template('home.html', data_points=data_points)
    except Exception as e:
        return render_template('error.html', error_message=str(e)), 500


@app.route('/add', methods=['GET', 'POST'])
def add_data_point():
    if request.method == 'POST':
        try:
            for param in ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']:
                if request.form.get(param) is None:
                    raise ValueError(f"Missing values.")

            sepal_length = float(request.form['sepal_length'])
            sepal_width = float(request.form['sepal_width'])
            petal_length = float(request.form['petal_length'])
            petal_width = float(request.form['petal_width'])
            species = int(request.form['species'])

            if any(val <= 0 for val in [sepal_length, sepal_width, petal_length, petal_width]):
                raise ValueError("Feature values must be positive numbers.")

            if species not in {0, 1, 2}:
                raise ValueError("The accepted values for species are: 0, 1, 2.")

            new_data_point = DataPoint(
                sepal_length=sepal_length,
                sepal_width=sepal_width,
                petal_length=petal_length,
                petal_width=petal_width,
                species=species
            )

            db.session.add(new_data_point)
            db.session.commit()
            return redirect(url_for('home'))

        except ValueError as ve:
            return render_template('error.html', error_message="Invalid data: " + str(ve)), 400
        except Exception as e:
            return render_template('error.html', error_message=str(e)), 400
    else:
        return render_template('add.html')


@app.route('/delete/<int:record_id>', methods=['POST'])
def delete_data_point(record_id):
    try:
        data_point = db.session.execute(
            db.select(DataPoint).where(DataPoint.id == record_id)).scalar_one_or_none()
        if data_point:
            db.session.delete(data_point)
            db.session.commit()
            return redirect(url_for('home'))
        else:
            return render_template('error.html', error_message="Record not found"), 404
    except Exception as e:
        return render_template('error.html', error_message=str(e)), 500


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            for param in ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']:
                if request.form.get(param) is None:
                    raise ValueError(f"Missing values.")

            sepal_length = float(request.form['sepal_length'])
            sepal_width = float(request.form['sepal_width'])
            petal_length = float(request.form['petal_length'])
            petal_width = float(request.form['petal_width'])

            if any(val <= 0 for val in [sepal_length, sepal_width, petal_length, petal_width]):
                raise ValueError("Feature values must be positive numbers.")

            predicted_species = make_prediction(sepal_length, sepal_width, petal_length, petal_width)

            return render_template('predict.html', prediction=predicted_species)

        except ValueError as ve:
            return render_template('error.html',
                                   error_message="Invalid data: "+str(ve)), 400
        except Exception as e:
            return render_template('error.html', error_message=str(e)), 400

    else:
        return render_template('predict.html')


# API methods

@app.route('/api/data', methods=['GET'])
def get_data_points_api():
    try:
        data_points = db.session.execute(db.select(DataPoint)).scalars().all()
        data_points_list = [
            {
                "id": dp.id,
                "sepal_length": dp.sepal_length,
                "sepal_width": dp.sepal_width,
                "petal_length": dp.petal_length,
                "petal_width": dp.petal_width,
                "species": dp.species
            }
            for dp in data_points
        ]
        return jsonify(data_points_list), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/data', methods=['POST'])
def add_data_point_api():
    try:
        data = request.get_json()

        required_keys = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
        if not all(key in data and data[key] is not None for key in required_keys):
            raise ValueError("Missing values.")

        sepal_length = float(data.get('sepal_length'))
        sepal_width = float(data.get('sepal_width'))
        petal_length = float(data.get('petal_length'))
        petal_width = float(data.get('petal_width'))
        species = int(data.get('species'))

        if any(val <= 0 for val in [sepal_length, sepal_width, petal_length, petal_width]):
            raise ValueError("Feature values must be positive numbers.")

        if species not in {0, 1, 2}:
            raise ValueError("The accepted values are: 0, 1, 2.")

        new_data_point = DataPoint(
            sepal_length=sepal_length,
            sepal_width=sepal_width,
            petal_length=petal_length,
            petal_width=petal_width,
            species=species
        )

        db.session.add(new_data_point)
        db.session.commit()

        return jsonify({"message": "Data point added", "id": new_data_point.id}), 201

    except ValueError as ve:
        return jsonify({"error": "Invalid data: " + str(ve)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/data/<int:record_id>', methods=['DELETE'])
def delete_data_point_api(record_id):
    try:
        data_point = db.session.execute(
            db.select(DataPoint).where(DataPoint.id == record_id)).scalar_one_or_none()
        if data_point:
            db.session.delete(data_point)
            db.session.commit()
            return jsonify({"message": "Record deleted", "record_id": record_id}), 200
        else:
            return jsonify({"error": "Record not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/predictions', methods=['GET'])
def api_predictions():
    try:
        if not all(request.args.get(param) for param in
                   ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']):
            raise ValueError("Missing values.")

        sepal_length = float(request.args.get('sepal_length'))
        sepal_width = float(request.args.get('sepal_width'))
        petal_length = float(request.args.get('petal_length'))
        petal_width = float(request.args.get('petal_width'))

        if any(val <= 0 for val in [sepal_length, sepal_width, petal_length, petal_width]):
            raise ValueError("Feature values must be positive numbers.")

        predicted_species = make_prediction(sepal_length, sepal_width, petal_length, petal_width)

        return jsonify({"prediction": predicted_species}), 200

    except ValueError as ve:
        return jsonify({"error": "Invalid data: " + str(ve)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# General methods

def make_prediction(sepal_length, sepal_width, petal_length, petal_width):
    data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    scaler = StandardScaler()

    data_points = db.session.execute(db.select(DataPoint)).scalars().all()

    if len(data_points) < 3:
        raise ValueError("There isn't enough data in the database to train the model. "
                         "Required at least 3 data points.")

    x_train = np.array([[dp.sepal_length, dp.sepal_width, dp.petal_length, dp.petal_width]
                        for dp in data_points])
    y_train = np.array([dp.species for dp in data_points])

    x_train_scaled = scaler.fit_transform(x_train)
    data_scaled = scaler.transform(data)

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(x_train_scaled, y_train)

    prediction = knn.predict(data_scaled)[0]

    species = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
    predicted_species = species[prediction]

    return predicted_species
