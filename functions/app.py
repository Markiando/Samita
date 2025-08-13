from flask import Flask, render_template,redirect,request,url_for
from callSamita import create_embedding, create_json_with_embeddings,find_closest_match, find_top_matches

app = Flask(__name__)
# inicio
@app.route("/", methods=['GET'])
def home():
    return render_template('index.html')
#Aqui aparece el top 3 
@app.route("/results", methods=['POST'])
#   
def showInfo():                                           #limpia el texto
    company_name = request.form.get("company_name")
    company_description = request.form.get("company_description")

    embedding = create_embedding("Nombre de la empresa: " + company_name + ". Giro de la empresa: " + company_description)
    #best_record,best_score = find_closest_match(embedding)
    #clase = best_record["clase"]
    #descripcion = best_record["descripcion"]
    #return render_template ('results.html', user_question=clase, descripcion= descripcion)
    results = find_top_matches(embedding)
    return render_template ('results.html', results=results)


@app.route("/create_json", methods=['GET'])
def createjson():                                           #limpia el texto
    create_json_with_embeddings()
    return "done"

#Hacer post a "/" para que reciba los datos del formulario
#def enviarform(request: datos):
    #response = callsamita(datos)3
    #return render_template('resultados.html')

if __name__=='__main__':
    app.run(debug=True)