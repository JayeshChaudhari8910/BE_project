from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
import numpy as np
import librosa
from sklearn.preprocessing import normalize
from keras.models import load_model
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'supersecretkey'

# Load the trained CNN model
model = load_model('practicenew.h5')

def init_sqlite_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                 username TEXT NOT NULL,
                 password TEXT NOT NULL)''')
    conn.commit()
    conn.close()

init_sqlite_db()

def preprocess_audio(file_path, sample_rate=22050, noise_threshold=0.05):
    signal, _ = librosa.load(file_path, sr=sample_rate)
    signal_normalized = normalize(signal.reshape(1, -1))
    return signal_normalized

def extract_features(file_path, mfcc=True, chroma=True, mel=True):
    signal_preprocessed = preprocess_audio(file_path)
    features = []
    if mfcc:
        mfccs = librosa.feature.mfcc(y=signal_preprocessed[0], n_mfcc=40)
        features.extend(np.mean(mfccs.T, axis=0))
    if chroma:
        chroma = librosa.feature.chroma_stft(y=signal_preprocessed[0])
        features.extend(np.mean(chroma.T, axis=0))
    if mel:
        mel_spec = librosa.feature.melspectrogram(y=signal_preprocessed[0])
        features.extend(np.mean(mel_spec.T, axis=0))
    return features

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        hashed_password = generate_password_hash(password, method='sha256')

        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
        conn.commit()
        conn.close()

        flash('You have successfully registered!', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username = ?", (username,))
        user = c.fetchone()
        conn.close()

        if user and check_password_hash(user[2], password):
            session['user_id'] = user[0]
            session['username'] = user[1]
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Login Unsuccessful. Please check username and password', 'danger')

    return render_template('login.html')

@app.route('/logout', methods=['POST'])
def logout():
    session.pop('user_id', None)
    session.pop('username', None)
    flash('You have been logged out.', 'success')
    return redirect(url_for('home'))

@app.route('/predict', methods=['POST'])
def predict():
    if 'user_id' not in session:
        flash('Please log in to use this feature.', 'danger')
        return redirect(url_for('login'))

    if request.method == 'POST':
        file = request.files['file']
        file.save('uploaded_audio.wav')
        features = extract_features('uploaded_audio.wav')
        features = np.array(features).reshape(1, -1)
        features = features.reshape(features.shape[0], features.shape[1], 1)
        prediction_prob = model.predict(features)
        predicted_class = np.argmax(prediction_prob, axis=1)
        emotion = ['Neutral', 'Angry', 'Happy', 'Fear', 'Sad'][predicted_class[0]]
        return render_template('result.html', emotion=emotion)

if __name__ == '__main__':
    app.run(debug=True)
