from flask import Flask, request, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import mysql.connector
from nltk.tokenize import MWETokenizer
from fuzzywuzzy import process
import numpy as np

app = Flask(__name__)


test_data ={
    "rumah sakit jakarta utara": [],
    "rumah sakit jakarta selatan": [],
    "rumah sakit jakarta barat": [],
    "rumah sakit jakarta pusat": [],
    "rumah sakit jakarta timur": [],
    "rumah sakit Cipete utara": ["Brawijaya"],
    "rumah sakit jakar utar": [],
    "rumah sakit jaka sel": [],
    "rumah sakit sekar": ["Sekar"],
    "rumah sakit Gdeong": ["Bina Waluya", "Tk.IV Cijantung Kesdam Jaya", "Pasar Rebo"],
    "rumah sakit Caw": ["Universitas Kristen Indonesia", "Budhi Asih "],
    "rumah sakit cik": ["PGI Cikini"],
    "tugu": ["Pelabuhan Jakarta", "Mulyasari"],
    "rumah sakit ibu dan anak jakarta utara": ["Family", "Grand Family", "Hermina Podomoro"],
    "rumah sakit bersalin jakarta selatan": ["Sofa Marwa", "Asih", "Panti Nugeraha", "Kartini", "Duren Tiga", "Avisena"]

}

# Konfigurasi database MySQL
db_config = {
    'user': 'root',
    'password': '',
    'host': 'localhost',
    'database': 'stbi',
    'port': '80'
}

#multi_word_expressions = []
_namaRumahSakit =[]
_jenistRumahSakit = []
_kelurahan = []
_kecamatan =[]
_kotaAdministrasi = []

tp = 0
fn = 0
fp = 0

allowed_keywords = []




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

        for item in df.to_dict('records'):
            if(item['kota_administrasi'] == "Jakarta Utara"):
                test_data["rumah sakit jakarta utara"].append(item['nama_rumah_sakit'])
                test_data["rumah sakit jakar utar"].append(item['nama_rumah_sakit'])
            elif(item['kota_administrasi'] == "Jakarta Selatan"):
                test_data["rumah sakit jakarta selatan"].append(item['nama_rumah_sakit'])
                test_data["rumah sakit jaka sel"].append(item['nama_rumah_sakit'])
            elif(item['kota_administrasi'] == "Jakarta Barat"):
                test_data["rumah sakit jakarta barat"].append(item['nama_rumah_sakit'])
            elif(item['kota_administrasi'] == "Jakarta Pusat"):
                test_data["rumah sakit jakarta pusat"].append(item['nama_rumah_sakit'])
            elif(item['kota_administrasi'] == "Jakarta Timur"):
                test_data["rumah sakit jakarta timur"].append(item['nama_rumah_sakit'])
        
    
        allowedWordList = []

        cursor.execute("SELECT DISTINCT nama_rumah_sakit FROM `data_rumah_sakit_di_dki_jakarta` WHERE nama_rumah_sakit != '-'")
        rows = cursor.fetchall()

        _namaRumahSakit  = set(pd.DataFrame(rows)['nama_rumah_sakit'].tolist()) 
        allowedWordList.append([word.lower() for string in _namaRumahSakit for word in string.split()])
        allowedWordList.append(pd.DataFrame(rows)['nama_rumah_sakit'].str.lower().tolist())

        cursor.execute("SELECT DISTINCT jenis_rumah_sakit FROM `data_rumah_sakit_di_dki_jakarta` WHERE jenis_rumah_sakit != '-'")
        rows = cursor.fetchall()

        _jenistRumahSakit  = set(pd.DataFrame(rows)['jenis_rumah_sakit'].tolist()) 
        allowedWordList.append([word.lower() for string in _jenistRumahSakit for word in string.split()])
        allowedWordList.append(pd.DataFrame(rows)['jenis_rumah_sakit'].str.lower().tolist())
        
        _jenistRumahSakit = [tuple(s.lower().split(" ")) for s in _jenistRumahSakit ] 


        cursor.execute("SELECT DISTINCT kota_administrasi FROM `data_rumah_sakit_di_dki_jakarta` WHERE kota_administrasi != '-'")
        rows = cursor.fetchall()

        _kotaAdministrasi  = set(pd.DataFrame(rows)['kota_administrasi'].tolist()) 
        allowedWordList.append([word.lower() for string in _kotaAdministrasi for word in string.split()])
        allowedWordList.append(pd.DataFrame(rows)['kota_administrasi'].str.lower().tolist())
        
        _kotaAdministrasi = [tuple(s.lower().split(" ")) for s in _kotaAdministrasi ] 
        
        
        cursor.execute("SELECT DISTINCT kelurahan FROM `data_rumah_sakit_di_dki_jakarta` WHERE kelurahan != '-'")
        rows = cursor.fetchall()

        _kelurahan = set(pd.DataFrame(rows)['kelurahan'].tolist()) 
        allowedWordList.append([word.lower() for string in _kelurahan for word in string.split()])
        _kelurahan = [tuple(s.lower().split(" ")) for s in _kelurahan]
        
    

        cursor.execute("SELECT DISTINCT kecamatan FROM `data_rumah_sakit_di_dki_jakarta` WHERE kecamatan != '-'")
        rows = cursor.fetchall()

        _kecamatan = set(pd.DataFrame(rows)['kecamatan'].tolist()) 
        allowedWordList.append([word.lower() for string in _kecamatan for word in string.split()])
        _kecamatan = [tuple(s.lower().split(" ")) for s in _kecamatan]
        

        allowed_keywords = [item for sublist in allowedWordList for item in sublist]
        allowed_keywords.append('rumah')
        allowed_keywords.append('sakit')

        return df, _kotaAdministrasi, _kelurahan, _kecamatan, allowed_keywords, _jenistRumahSakit

    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return pd.DataFrame()  # Kembalikan DataFrame kosong jika terjadi error

    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()
# Inisialisasi dan fit TF-IDF Vectorizer
df, _kotaAdministrasi, _kelurahan, _kecamatan, allowed_keywords, _jenistRumahSakit = load_data_from_mysql()

if not df.empty:
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['search_data'])

kota_tokenizer = MWETokenizer(_kotaAdministrasi, separator=' ')
def Evaluation(query, expected, result):
    global tp 
    global fn
    global fp 
    for item in result:
        if(item in expected):
            tp +=1
            expected.remove(item)
        else:
            fp +=1
    fn += len(expected)
def correct_typos(query, allowed, limit=1):
    corrected_query = []
    for word in query.split():
        # Cari kata yang paling mirip, dengan skor minimal 80
        result = process.extractOne(word, allowed_keywords, score_cutoff=50)
        if result:  # Cek apakah ada hasil yang ditemukan
            corrected_word, score = result  # Unpacking karena result bukan None
            corrected_query.append(corrected_word)
        else:
            corrected_query.append(word)  # Jika tidak ada yang cukup mirip, gunakan kata asli
    
    return ' '.join(corrected_query)

@app.route('/', methods=['GET', 'POST'])
def search():
    results = []
    keylist = list(test_data.keys())
    for item in keylist:
        Evaluation(item, test_data[item], GetResults(item, True))
    precision = GetPrecision()
    recall = GetRecall()
    f1score = GetF1Score()

    if request.method == 'POST' and not df.empty:
        original_query = request.form['query']
        corrected_query = correct_typos(original_query, allowed_keywords)  # Koreksi typo
        results = GetResults(corrected_query, False);

        return render_template('index.html', results=results.to_dict('records'), query=original_query, corrected_query=corrected_query, precision = precision, recall = recall, f1score = f1score)
    return render_template('index.html', results=None, precision = precision, recall = recall, f1score = f1score)

def GetResults(corrected_query, isEvaluating):
    kota_tokens = kota_tokenizer.tokenize(corrected_query.split())  # Gunakan query yang sudah dikoreksi
        
        
    kota = [token for token in kota_tokens if token.lower() in [keyword.lower() for keyword in [' '.join(tup) for tup in _kotaAdministrasi]]]


    query_vector = tfidf_vectorizer.transform([corrected_query])  # Gunakan query yang sudah dikoreksi
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[::-1]
        
    filtered_indices = [i for i in top_indices if similarities[i] > 0.1]

        

    if kota:
        finalresults = df.iloc[filtered_indices][df['kota_administrasi'].str.contains('|'.join(kota), case=False, na=False)]
    else:
        finalresults = df.iloc[filtered_indices]
      
    dictResults = finalresults.to_dict('records')
    resultRumahSakit = [item['nama_rumah_sakit'] for item in dictResults]

    if(isEvaluating):
        return resultRumahSakit
    else:
        return finalresults
        

def GetPrecision():
    global tp
    global fp
    return (tp/(tp+fp))

def GetRecall():
    global tp
    global fn
    return (tp/(tp+fn))

def GetF1Score():
    precision = GetPrecision()
    recall = GetRecall()
    return 2*precision*recall / (precision+recall)


if __name__ == '__main__':
    app.run(debug=True)