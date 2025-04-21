from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load model, encoders, and feature column order
model = joblib.load('student_grade_predictor.joblib')
label_encoders = joblib.load('label_encoders.joblib')
feature_columns = joblib.load('feature_columns.joblib')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    error = None

    if request.method == 'POST':
        try:
            input_data = {
                'school': request.form['school'],
                'sex': request.form['sex'],
                'address': request.form['address'],
                'famsize': request.form['famsize'],
                'Pstatus': request.form['Pstatus'],
                'studytime': int(request.form['studytime']),
                'failures': int(request.form['failures']),
                'absences': int(request.form['absences']),
            }

            # Encode categorical variables
            for col in input_data:
                if col in label_encoders:
                    le = label_encoders[col]
                    if input_data[col] in le.classes_:
                        input_data[col] = le.transform([input_data[col]])[0]
                    else:
                        raise ValueError(f"Invalid input for '{col}': {input_data[col]}")

            # Convert to DataFrame and reorder columns
            df = pd.DataFrame([input_data])
            df = df.reindex(columns=feature_columns)

            # Predict
            prediction = model.predict(df)[0]

        except ValueError as ve:
            error = f"Error: {str(ve)}"
        except Exception as e:
            error = f"Something went wrong: {str(e)}"

    return render_template('index.html', prediction=prediction, error=error)

if __name__ == '__main__':
    app.run(debug=True)
