# https://github.com/computervisioneng/pneumonia-classification-web-app-python-streamlit

import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
FILES_IMAGES = PROJECT_ROOT / "images" / "normal"
FILES_MODELS = PROJECT_ROOT / "models"

# set title
st.title('Pneumonia classification')

# set header
st.header('Please upload a chest X-ray image')

# upload file
file = st.file_uploader(FILES_IMAGES, type=['jpeg', 'jpg', 'png'])

# Verificar se o arquivo do modelo existe
model_path = FILES_MODELS
if not os.path.exists(model_path):
    st.error(f"❌ Arquivo do modelo não encontrado em: {model_path}")
    st.stop()

# load classifier with error handling
try:
    model = load_model(model_path)
    st.success("✅ Modelo carregado com sucesso!")
except Exception as e:
    st.error(f"❌ Erro ao carregar o modelo: {str(e)}")
    st.info("💡 Possíveis soluções:")
    st.info("1. Verifique se o arquivo do modelo não está corrompido")
    st.info("2. Verifique a compatibilidade da versão do TensorFlow/Keras")
    st.info("3. Recrie o modelo se necessário")
    st.stop()

# load class names
try:
    with open(FILES_MODELS + '/labels.txt', 'r') as f:
        class_names = [a[:-1].split(' ')[1] for a in f.readlines()]
    st.success(f"✅ Classes carregadas: {class_names}")
except Exception as e:
    st.error(f"❌ Erro ao carregar labels: {str(e)}")
    # Fallback para classes padrão
    class_names = ['Normal', 'Pneumonia']
    st.info(f"🔧 Usando classes padrão: {class_names}")

# display image and classify
if file is not None:
    try:
        image = Image.open(file).convert('RGB')
        st.image(image, use_column_width=True)

        # classify image
        class_name, conf_score = classify(image, model, class_names)

        # write classification
        st.write("## {}".format(class_name))
        st.write("### score: {}%".format(int(conf_score * 1000) / 10))
        
    except Exception as e:
        st.error(f"❌ Erro ao processar imagem: {str(e)}")



