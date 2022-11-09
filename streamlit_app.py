import streamlit as st
import nltk
import joblib
import tensorflow as tf
from keras.utils import pad_sequences
from tensorflow.python.keras import layers
import pandas as pd
# from keras_nlp import TransformerEncoder, TransformerDecoder, TokenAndPositionEmbedding
import spacy
import string
from calendar_view.calendar import Calendar
from calendar_view.core.event import EventStyles
from calendar_view.config import style
from calendar_view.core import data
from calendar_view.core.event import Event
from datetime import datetime, timedelta, time, date
from random import randint
import numpy as np
import calendar as cal
import plotly.express as px

st.set_page_config(
    page_title="Printer chatbot",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

if 'events' not in st.session_state:
    st.session_state.events = []

if 'current_day' not in st.session_state:
    st.session_state.current_day = date.today()

if 'last_hour' not in st.session_state:
    st.session_state.last_hour = time()  # minuit

if 'begin' not in st.session_state:
    st.session_state.begin = time()  # minuit par defaut

if 'fin_matin' not in st.session_state:
    st.session_state.fin_matin = time(hour=12)

if 'debut_aprem' not in st.session_state:
    st.session_state.debut_aprem = time(hour=12)

if 'fin_aprem' not in st.session_state:
    st.session_state.fin_aprem = time(hour=23, minute=00)

if 'dates' not in st.session_state:
    st.session_state.dates = '{} - {}'.format((cal.day_name[st.session_state.current_day.weekday()])[:2],
                                              (cal.day_name[
                                                  (st.session_state.current_day - timedelta(days=1)).weekday()])[:2])

if 'color' not in st.session_state:
    st.session_state.color = [EventStyles.GREEN,
                              EventStyles.GRAY,
                              EventStyles.BLUE]

if 'hour_height' not in st.session_state:
    st.session_state.hour_height = 50

if 'fig' not in st.session_state:
    st.session_state.fig = 'none'

nltk.download('stopwords')
nltk.download('punkt')


@st.cache(allow_output_mutation=True)
def spacy_load(model_name):
    ret = spacy.load(model_name, disable=["parser", "ner"])
    return ret


nlp = spacy_load('fr_core_news_sm')
#nlp = fr_core_news_sm.load(disable=["parser", "ner"])

stopwords = list(string.punctuation) + ['-', "'", " ", "  ", "   "] # + ntlk.corpus.stopwords.words('french')


def transform_text(text: str):
    return [(word.lemma_, word.pos_) for word in nlp(text) if word.text not in stopwords]


def tokenizer(x):
    return [ret[0] for ret in transform_text(x)]


@st.cache(allow_output_mutation=True)
def joblib_load(model_name):
    model = joblib.load(model_name)
    return (model)


@st.cache(allow_output_mutation=True)
def keras_load(model_path):
    model = tf.keras.models.load_model(model_path)
    return (model)


def padding(lst):
    return pad_sequences(lst, maxlen=32, padding='post').tolist()


# voir variabl max_len et padding dans le dernier notebook pour les param TODO


def to_numeric(roles):
    roles_map = {'unk': 0,
                 'ADJ': 1,
                 'ADP': 2,
                 'ADV': 3,
                 'AUX': 4,
                 'CCONJ': 5,
                 'DET': 6,
                 'INTJ': 7,
                 'NOUN': 8,
                 'NUM': 9,
                 'PRON': 10,
                 'PROPN': 11,
                 'PUNCT': 12,
                 'SCONJ': 13,
                 'SPACE': 14,
                 'VERB': 15,
                 'X': 16}
    ret = [roles_map[r] for r in roles]
    return ret


def parse_raw(phrase):
    parsed = transform_text(phrase)
    tokens = [t[0] for t in parsed]
    roles = [t[1] for t in parsed]
    ph_tk = " ".join([tk for tk in tokens])
    points = [(0 if "." not in tk else 1) for tk in tokens]
    roles = to_numeric(roles)
    return ph_tk, points, roles


model_label = joblib_load("new_models/label_model.joblib")
model_missdoc = joblib_load("models/missing_document_pipe.joblib")
model_misspage = joblib_load("models/missing_pages_pipe.joblib")
model_doc = keras_load("new_models/trans_document.keras.file")
model_num = keras_load("new_models/trans_pages.keras.file")
model_missdoc_keras = keras_load("new_models/trans_md.keras.file")
model_misspage_keras = keras_load("new_models/trans_mp.keras.file")
model_label_keras = keras_load("new_models/trans_lb.keras.file")


def pred_binaire(phrase, thresh, model):
    data = pd.DataFrame()
    data['phrase'] = [phrase]

    result = {}
    result['score'] = model.decision_function(data)[0]
    result['proba'] = model.predict_proba(data)[:, 1][0]
    result['decision'] = result['score'] > thresh
    return result


def extract_name_page(ph_tk, pts, roles):
    pts_pd = padding([pts])
    roles_pd = padding([roles])
    ph_tka = np.array([ph_tk])
    preds_doc = model_doc.predict([ph_tka, np.stack(pts_pd), np.stack(roles_pd)], verbose=0)
    idx_doc = preds_doc.argmax()

    preds_pages = model_num.predict([ph_tka, np.stack(roles_pd)], verbose=0)
    idx_pages = preds_pages.argmax()

    return [ph_tk.split(' ')[idx_doc], ph_tk.split(' ')[idx_pages]], preds_pages, preds_doc


def missing_doc(ph_tk, pts, roles):
    pts_pd = padding([pts])
    roles_pd = padding([roles])
    ph_tka = np.array([ph_tk])
    preds_doc = model_missdoc_keras.predict([ph_tka, np.stack(pts_pd), np.stack(roles_pd)], verbose=0)
    return preds_doc.tolist()[0][0]


def label_mp_keras_pred(ph_tk, pts, roles, model):
    roles_pd = padding([roles])
    ph_tka = np.array([ph_tk])
    pred = model.predict([ph_tka, np.stack(roles_pd)], verbose=0)
    return pred.tolist()[0][0]


def create_red_events():
    nb_day_of_week = 7
    day = st.session_state.current_day
    for i in range(nb_day_of_week):
        st.session_state.events.append(
            Event(
                title='Pause',
                day=day,
                start=st.session_state.fin_matin,
                end=st.session_state.debut_aprem,
                style=EventStyles.RED
            )
        )
        day = day + timedelta(days=1)



def make_df(phrase_tk, tableau_doc, tableau_pages):
    lines = []
    for i in range(len(phrase_tk)):
        lines.append([phrase_tk[i],tableau_doc[i], 'document'])
        lines.append([phrase_tk[i],tableau_pages[i], 'pages'])

    return pd.DataFrame(lines, columns = ['token','val', 'cat'])

def hist(phrase_tk, tableau_doc, tableau_pages):
  """
    phrase_tk = tableau de longueur N de tous les tokens (coucou ça va ? -> ["coucou", "ça", "va"])
    tableau_xxx = tableau de 32 flottants entre 0 et 1 (somme à 1) [0.1, 0.2, ....]
  """
  df = make_df(phrase_tk, tableau_doc, tableau_pages)

  fig = px.histogram(df, x= df.token, y= df.val, 
                     labels={'x':'phrase', 'y':'scores extraction', 'color' :'Type'}, 
                     color='cat', 
                     title = "Scores d'extraction pour chaque mot de la phrase"
                    )
  fig.update_layout(barmode='group')
  return fig


def click_submit(req, form, events, label_thresh, keras_md_thresh, keras_mp_thresh):
    # label = pred_binaire(req, label_thresh, model_label)
    # d'abbord parser la phrase puis traiter pour gagner en perf
    ph_tk, pts, roles = parse_raw(req.lower())
    label = label_mp_keras_pred(ph_tk, pts, roles, model_label_keras)
    if label > label_thresh:
        form.write('la requête est invalide, veuillez réessayer')

    if label < label_thresh:

        if missing_doc(ph_tk, pts, roles) > keras_md_thresh:
            form.write("Le nom du document est manquant, veuillez le rajouter")
            return
        if label_mp_keras_pred(ph_tk, pts, roles, model_misspage_keras) > keras_mp_thresh:
            form.write("Le nombre de pages du document est manquant, veuillez le rajouter")
            return

        form.write("requête acceptée")
        token, pred_page, pred_doc = extract_name_page(ph_tk, pts, roles)
        sec_duration = int(token[1])
        if sec_duration < 60:
            sec_duration = 60
        if sec_duration > 14400:
            form.write("Trop de pages à imprimer, vous ne pouvez pas imprimer plus de 14400 pages pour des soucis de "
                       "partage du matériel.")
            return

        name = token[0]
        end_time = (datetime.combine(date.today(), st.session_state.last_hour) + timedelta(seconds=sec_duration)).time()
        color = st.session_state.color[randint(0, len(st.session_state.color) - 1)]

        #        if end_time < st.session_state.last_hour:
        #            # cas ou on est sur un nouveau jour
        #            events.append(
        #                Event(
        #                    title=name,
        #                    day=st.session_state.current_day,
        #                    start=st.session_state.last_hour,
        #                    end='23:59',
        #                    style=EventStyles.GREEN
        #                )
        #            )
        #            st.session_state.current_day = st.session_state.current_day + timedelta(days=1)
        #            events.append(
        #                Event(
        #                    title=name,
        #                    day=st.session_state.current_day,
        #                    start='00:00',
        #                    end=end_time,
        #                    style=EventStyles.GREEN
        #                )
        #            )
        #        else:  # cas "normal"
        #            events.append(
        #                Event(title=name
        #                      , day=st.session_state.current_day
        #                      , start=st.session_state.last_hour
        #                      , end=end_time
        #                      , style=EventStyles.GREEN
        #                      )
        #            )
        #        st.session_state.last_hour = end_time
        ###### nouvelle implem avec les bornes#####
        if (st.session_state.last_hour >= st.session_state.begin) & (
                st.session_state.last_hour < st.session_state.fin_matin):
            # la derniere impression a eu lieu le matin :
            if end_time >= st.session_state.fin_matin:
                events.append(
                    Event(
                        title=name,
                        day=st.session_state.current_day,
                        start=st.session_state.last_hour,
                        end=st.session_state.fin_matin,
                        style=color
                    )
                )
                fin_mat = timedelta(hours=st.session_state.fin_matin.hour, minutes=st.session_state.fin_matin.minute,
                                    seconds=st.session_state.fin_matin.second)
                last_hou = timedelta(hours=st.session_state.last_hour.hour, minutes=st.session_state.last_hour.minute,
                                     seconds=st.session_state.last_hour.second)
                temps_restant = timedelta(seconds=sec_duration) - (fin_mat - last_hou)
                heure_finale = (datetime.combine(date.today(), st.session_state.debut_aprem) + timedelta(
                    seconds=temps_restant.total_seconds())).time()
                events.append(
                    Event(
                        title=name,
                        day=st.session_state.current_day,
                        start=st.session_state.debut_aprem,
                        end=heure_finale,
                        style=color
                    )
                )
                st.session_state.last_hour = heure_finale
            else:
                events.append(
                    Event(title=name
                          , day=st.session_state.current_day
                          , start=st.session_state.last_hour
                          , end=end_time
                          , style=color
                          )
                )
                st.session_state.last_hour = end_time
        elif (st.session_state.last_hour >= st.session_state.debut_aprem) & (
                st.session_state.last_hour <= st.session_state.fin_aprem):
            if (end_time > st.session_state.fin_aprem) | (end_time < st.session_state.debut_aprem):
                # si il est apres la fin de journée (on recommence le matin
                events.append(
                    Event(title=name
                          , day=st.session_state.current_day
                          , start=st.session_state.last_hour
                          , end=st.session_state.fin_aprem
                          , style=color
                          )
                )
                st.session_state.current_day = st.session_state.current_day + timedelta(days=1)
                fin_jour = timedelta(hours=st.session_state.fin_aprem.hour, minutes=st.session_state.fin_aprem.minute,
                                     seconds=st.session_state.fin_aprem.second)
                last_hou = timedelta(hours=st.session_state.last_hour.hour, minutes=st.session_state.last_hour.minute,
                                     seconds=st.session_state.last_hour.second)

                temps_restant = timedelta(seconds=sec_duration) - (fin_jour - last_hou)
                heure_finale = (datetime.combine(date.today(), st.session_state.begin) + timedelta(
                    seconds=temps_restant.total_seconds())).time()
                events.append(
                    Event(title=name
                          , day=st.session_state.current_day
                          , start=st.session_state.begin
                          , end=heure_finale
                          , style=color
                          )
                )
                st.session_state.last_hour = heure_finale
            else:
                events.append(
                    Event(title=name
                          , day=st.session_state.current_day
                          , start=st.session_state.last_hour
                          , end=end_time
                          , style=color
                          )
                )
                st.session_state.last_hour = end_time
        tmp = ph_tk.split(' ')
        st.session_state.fig = hist(tmp, pred_doc[0], pred_page[0])



def main():
    label_thresh = 1.05
    keras_label_thresh = 0.5
    keras_missing_pages_thresh = 0.5
    missing_doc_thresh = 1.79
    keras_missing_doc_thresh = 0.67

    with st.sidebar:
        form = st.form("submissionForm")
        with form:
            request = st.text_input(
                "Poser votre requête d'impression ici : ",
                value="",
                placeholder="ex: Bonjour, j'aimerais imprimer 12 copies du doc2"
            )
            validation = st.form_submit_button("Envoyer la requête")

        if validation:
            # faire la modification du calendrier
            click_submit(request, form, st.session_state.events,
                         keras_label_thresh,
                         keras_missing_doc_thresh,
                         keras_missing_pages_thresh)

        st.write("Plage horaire du chatbot ")
        st.write("!! Format = 'HH:mm' !!")
        matin_debut = st.text_input(
            "Heure de début",
            placeholder="par defaut : 00:00"
        )
        matin_fin = st.text_input(
            "Heure de fin de matinée",
            placeholder="par defaut : 12:00"
        )
        aprem_debut = st.text_input(
            "Heure de début d'après-midi",
            placeholder="par defaut : 12:00"
        )
        aprem_fin = st.text_input(
            "Heure de fin",
            placeholder="par defaut : 23:00"
        )
        st.write("Toutes modifications de la plage horaire entraineront la perte des évènements précedents")
        if st.button("Changer la plage horaire"):
            st.session_state.events = []
            if matin_debut != "":
                heure, mymin = matin_debut.split(':')
                st.session_state.begin = time(hour=int(heure), minute=int(mymin))
            if matin_fin != "":
                heure, mymin = matin_fin.split(':')
                st.session_state.fin_matin = time(hour=int(heure), minute=int(mymin))
            if aprem_debut != "":
                heure, mymin = aprem_debut.split(':')
                st.session_state.debut_aprem = time(hour=int(heure), minute=int(mymin))
            if aprem_fin != "":
                heure, mymin = aprem_fin.split(':')
                st.session_state.fin_aprem = time(hour=int(heure), minute=int(mymin))

            st.session_state.last_hour = st.session_state.begin
            st.session_state.current_day = date.today()
            create_red_events()

    st.header("Voici les horaires disponibles dans l'agenda de la photocopieuse")
    st.session_state.hour_height = st.slider(label='distance entre 2 heures', min_value=40, max_value=200, value=50)
    style.hour_height = st.session_state.hour_height
    calendar = Calendar.build(config)
    calendar.add_events(st.session_state.events)
    calendar.save("calendar.png")
    st.image(calendar.full_image)

    if st.session_state.fig != 'none':
        st.plotly_chart(st.session_state.fig)


if __name__ == "__main__":
    # on cale le calendrier sur le jour d'aujourd'hui
    dates = st.session_state.dates
    hours_schedule = '{} - {}'.format(int(st.session_state.begin.hour), int(st.session_state.fin_aprem.hour))
    config = data.CalendarConfig(
        lang='en',
        title='Schedule',
        dates=dates,
        hours=hours_schedule,
        show_year=True,
        title_vertical_align='top'
    )

    main()
