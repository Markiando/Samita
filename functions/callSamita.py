from clients.openai_client import openai_client
import csv, json
from scipy.spatial.distance import cosine 
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity



def create_embedding(question: str): #tipo str
    # Convierte texto a embedding
    response = openai_client.embeddings.create(
        input=question,
        model="text-embedding-3-large" 
    )
    return response.data[0].embedding #envia un json
# Tenemos una lista de todas las clases de IMPI, las cuales convertimos en embedding
# y guardamos los resultados en un archivo json
def create_json_with_embeddings():
# Ruta de entrada y salida
    csv_file = "data\clasificacion_niza_clase_descripcion_v3.csv"   # Tu archivo CSV con columnas: clase, descripcion
    json_file = "data\results.json"

    resultados = []
    # Leer CSV y procesar
    with open(csv_file, mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
                clase = row["clase"]
                descripcion = row["descripcion"]
                embedding = create_embedding(descripcion)

                # Guardar en el formato deseado
                resultados.append({
                    "clase": clase,
                    "descripcion": descripcion,
                    "embedding": embedding
                })

    # Guardar resultados en un JSON
    with open("results.json", mode="w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=4)

    print(f"Procesado completo. {len(resultados)} registros guardados en {json_file}")
#     Compara la descripcion de la marca del usuario,
#     con los vectores del archivo json
def find_top_matches(question_embedding, top_n=3):
    # Lo carga una vez
    with open("results.json", "r", encoding="utf-8") as f:
        RECORDS = json.load(f)

    # Calcula la similaridad de cada registro
    scored_records = []
    for record in RECORDS:
        similarity = cosine_similarity(
            np.array(record["embedding"]).reshape(1, -1),
            np.array(question_embedding).reshape(1, -1)
        )[0][0]
        scored_records.append((record, similarity))

    # Ordena la similaridad
    scored_records.sort(key=lambda x: x[1], reverse=True)

    # Regresa un top 3 de recomendaciones
    return scored_records[:top_n]


#def find_closest_match(question_embedding):
   # best_score = -1
   # best_record = None

    # Load once at startup
    #with open("results.json", "r", encoding="utf-8") as f:
    #    RECORDS = json.load(f)
   # for record in RECORDS:
        #similarity = cosine(question_embedding, record["embedding"])
        #similarity  = cosine_similarity(np.array(record["embedding"]).reshape(1, -1), np.array(question_embedding).reshape(1, -1))[0][0]
       # if similarity > best_score:
      #      best_score = similarity
     #       best_record = record

    #return best_record, best_score
    # return best_record, best_score

