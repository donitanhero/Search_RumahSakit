from flask import Flask, request, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import mysql.connector
import nltk
from nltk.tokenize import MWETokenizer

app = Flask(__name__)

# Konfigurasi database MySQL
db_config = {
    'user': 'root',
    'password': '',
    'host': 'localhost',
    'database': 'stbi',
    'port': '80'
}

# Definisikan frasa yang ingin dijadikan satu token
multi_word_expressions = [
    ('jakarta', 'utara'),
    ('jakarta', 'selatan'),
    ('jakarta', 'barat'),
    ('jakarta', 'timur'),
    ('jakarta', 'pusat')
]

tokenizer = MWETokenizer(multi_word_expressions, separator=' ')


# Fungsi untuk mengambil data dari MySQL
def load_data_from_mysql():
    try:
        conn = mysql.connector.connect(
        host="localhost",
        user="root",
        passwd="",
        database="stbi",
        port=3306
        )
        cursor = conn.cursor(dictionary=True)

        # Query untuk mengambil semua data dari tabel rumah_sakit_detail
        cursor.execute('SELECT * FROM data_rumah_sakit_di_dki_jakarta')
        rows = cursor.fetchall()

        # Konversi ke DataFrame
        df = pd.DataFrame(rows)

        # Menggabungkan kolom relevan untuk pencarian
        df['search_data'] = df.apply(lambda row: f"{row['kode_rumah_sakit']} {row['nama_rumah_sakit']} {row['jenis_rumah_sakit']} {row['alamat_rumah_sakit']} {row['kelurahan']} {row['kecamatan']} {row['kota_administrasi']} {row['kode_pos']} {row['nomor_telepon']} {row['nomor_fax']} {row['website']} {row['email']} {row['telepon_humas']} {row['website']}", axis=1)

        return df

    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return pd.DataFrame()  # Kembalikan DataFrame kosong jika terjadi error

    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

# Inisialisasi dan fit TF-IDF Vectorizer
df = load_data_from_mysql()
if not df.empty:
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['search_data'])

@app.route('/', methods=['GET', 'POST'])
def search():
    results = []
    if request.method == 'POST' and not df.empty:
        query = request.form['query'].lower()
        
        # Tokenisasi query dengan mempertimbangkan multi-word expressions
        tokens = tokenizer.tokenize(query.split())
        
        # Asumsikan lokasi spesifik dapat dikenali sebagai satu token
        locations = [token for token in tokens if token in ['jakarta utara', 'jakarta selatan', 'jakarta barat', 'jakarta timur', 'jakarta pusat']]
        
        query_vector = tfidf_vectorizer.transform([' '.join(tokens)])  # Gunakan tokens yang sudah ditokenisasi
        similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
        top_indices = similarities.argsort()[::-1]
         # Filter berdasarkan nilai kesamaan > 0.1
        filtered_indices = [i for i in top_indices if similarities[i] > 0.1]

        if locations:  # Jika ada kata kunci lokasi yang dikenali, filter hasil berdasarkan kata kunci
            results = df.iloc[filtered_indices][df['kota_administrasi'].str.contains('|'.join(locations), case=False, na=False)]
        else:  # Jika tidak ada kata kunci lokasi, gunakan skor TF-IDF yang telah difilter
            results = df.iloc[filtered_indices]

        return render_template('index.html', results=results.to_dict('records'), query=query)
    return render_template('index.html', results=None)

if __name__ == '__main__':
    app.run(debug=True)