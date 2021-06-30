# -*- coding: utf-8 -*-

from flask import Flask, render_template, request, redirect, session, flash, jsonify
from mysqlconnection import MySQLConnector
import pandas as pd

# from __future__ import print_function
import re
import os
import json
import MySQLdb
import MySQLdb.cursors as cursors

import numpy as np
import pandas as pd


from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np
import threading
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras import backend as K


histarray={'dengue':0, 'maleria':0, 'normal': 0}

def load_model():
    try:
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        model.load_weights("weights.hdf5")
        print("Model successfully loaded from disk.")
        
        #compile again
        model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        return model
    except Exception as e:
        print (e)
        print("""Model not found. Please train the CNN by running the script """)
        return None
    
    
def update(histarray2):
    global histarray
    histarray=histarray2



l1=['back_pain','constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine',
'yellowing_of_eyes','acute_liver_failure','fluid_overload','swelling_of_stomach',
'swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','throat_irritation',
'redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs',
'fast_heart_rate','pain_during_bowel_movements','pain_in_anal_region','bloody_stool',
'irritation_in_anus','neck_pain','dizziness','cramps','bruising','obesity','swollen_legs',
'swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails',
'swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips',
'slurred_speech','knee_pain','hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints',
'movement_stiffness','spinning_movements','loss_of_balance','unsteadiness',
'weakness_of_one_body_side','loss_of_smell','bladder_discomfort','foul_smell_of urine',
'continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)',
'depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain',
'abnormal_menstruation','dischromic _patches','watering_from_eyes','increased_appetite','polyuria','family_history','mucoid_sputum',
'rusty_sputum','lack_of_concentration','visual_disturbances','receiving_blood_transfusion',
'receiving_unsterile_injections','coma','stomach_bleeding','distention_of_abdomen',
'history_of_alcohol_consumption','fluid_overload','blood_in_sputum','prominent_veins_on_calf',
'palpitations','painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling',
'silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose',
'yellow_crust_ooze']

disease=['Fungal infection','Allergy','GERD','Chronic cholestasis','Drug Reaction',
'Peptic ulcer diseae','AIDS','Diabetes','Gastroenteritis','Bronchial Asthma','Hypertension',
' Migraine','Cervical spondylosis',
'Paralysis (brain hemorrhage)','Jaundice','Malaria','Chicken pox','Dengue','Typhoid','hepatitis A',
'Hepatitis B','Hepatitis C','Hepatitis D','Hepatitis E','Alcoholic hepatitis','Tuberculosis',
'Common Cold','Pneumonia','Dimorphic hemmorhoids(piles)',
'Heartattack','Varicoseveins','Hypothyroidism','Hyperthyroidism','Hypoglycemia','Osteoarthristis',
'Arthritis','(vertigo) Paroymsal  Positional Vertigo','Acne','Urinary tract infection','Psoriasis',
'Impetigo']

l2=[]
for x in range(0,len(l1)):
    l2.append(0)

# TESTING DATA df 
df=pd.read_csv("Training.csv")

df.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
'Migraine':11,'Cervical spondylosis':12,
'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
'(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
'Impetigo':40}},inplace=True)

# print(df.head())

X= df[l1]

y = df[["prognosis"]]
np.ravel(y)
# print(y)

# Testing DATA tr
tr=pd.read_csv("Testing.csv")
tr.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
'Migraine':11,'Cervical spondylosis':12,
'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
'(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
'Impetigo':40}},inplace=True)

X_test= tr[l1]
y_test = tr[["prognosis"]]
np.ravel(y_test)



def DecisionTree(Symptom1,Symptom2,Symptom3,Symptom4,Symptom5):

    from sklearn import tree

    clf3 = tree.DecisionTreeClassifier()   # empty model of the decision tree
    clf3 = clf3.fit(X,y)

    # calculating accuracy
    from sklearn.metrics import accuracy_score
    y_pred=clf3.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    session['dtacc']= accuracy_score(y_test, y_pred)
    print(accuracy_score(y_test, y_pred,normalize=False))
    
    psymptoms = [Symptom1,Symptom2,Symptom3,Symptom4,Symptom5]

    for k in range(0,len(l1)):
        # print (k,)
        for z in psymptoms:
            if(z==l1[k]):
                l2[k]=1

    inputtest = [l2]
    predict = clf3.predict(inputtest)
    predicted=predict[0]

    h='no'
    for a in range(0,len(disease)):
        if(predicted == a):
            h='yes'
            break


    if (h=='yes'):
        disd=disease[a]
    else:
        disd="Not Found"

    return disd


def randomforest(Symptom1,Symptom2,Symptom3,Symptom4,Symptom5):
    from sklearn.ensemble import RandomForestClassifier
    clf4 = RandomForestClassifier()
    clf4 = clf4.fit(X,np.ravel(y))

    # calculating accuracy-------------------------------------------------------------------
    from sklearn.metrics import accuracy_score
    y_pred=clf4.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    session['rfacc']= accuracy_score(y_test, y_pred)
    print(accuracy_score(y_test, y_pred,normalize=False))

    psymptoms = [Symptom1,Symptom2,Symptom3,Symptom4,Symptom5]

    for k in range(0,len(l1)):
        for z in psymptoms:
            if(z==l1[k]):
                l2[k]=1

    inputtest = [l2]
    predict = clf4.predict(inputtest)
    predicted=predict[0]

    h='no'
    for a in range(0,len(disease)):
        if(predicted == a):
            h='yes'
            break

    if (h=='yes'):
       disr=disease[a]
    else:
        disr="Not Found"
    return disr


def NaiveBayes(Symptom1,Symptom2,Symptom3,Symptom4,Symptom5):
    from sklearn.naive_bayes import GaussianNB
    gnb = GaussianNB()
    gnb=gnb.fit(X,np.ravel(y))

    # calculating accuracy-------------------------------------------------------------------
    from sklearn.metrics import accuracy_score
    y_pred=gnb.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    session['nbacc']= accuracy_score(y_test, y_pred)
    print(accuracy_score(y_test, y_pred,normalize=False))

    psymptoms = [Symptom1,Symptom2,Symptom3,Symptom4,Symptom5]
    for k in range(0,len(l1)):
        for z in psymptoms:
            if(z==l1[k]):
                l2[k]=1

    inputtest = [l2]
    predict = gnb.predict(inputtest)
    predicted=predict[0]

    h='no'
    for a in range(0,len(disease)):
        if(predicted == a):
            h='yes'
            break

    if (h=='yes'):
        disn=disease[a]
    else:
        disn="Not Found"

    return disn



EMAIL_REGEX = re.compile(r'^[a-zA-Z0-9.+_-]+@[a-zA-Z0-9._-]+\.[a-zA-Z]+$')
NAME_REGEX = re.compile(r'[0-9]')
PASS_REGEX = re.compile(r'.*[A-Z].*[0-9]')

app = Flask(__name__)
app.secret_key= 'disease'
mysql = MySQLConnector(app, 'twitter_data')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/UserLogin')
def user():
    return render_template('UserLogin.html')


@app.route('/Register')
def register():
    return render_template('Register.html')


@app.route('/logout')
def logout():
    return render_template('index.html')


@app.route('/Prediction', methods=['POST'])
def Prediction():
    symp1=request.form['symp1']
    symp2=request.form['symp2']
    symp3=request.form['symp3']
    symp4=request.form['symp4']
    symp5=request.form['symp5']
    pic=request.form['pic']
    print("name ",pic)
    '''
    session['symp1']= symp1
    session['symp2']= symp2
    session['symp3']= symp3
    session['symp4']= symp4
    session['symp5']= symp5
    '''
    disdt=DecisionTree(symp1,symp2,symp3,symp4,symp5)
    disrf=randomforest(symp1,symp2,symp3,symp4,symp5)
    disnb=NaiveBayes(symp1,symp2,symp3,symp4,symp5)

    session['disdt']= disdt
    session['disrf']= disrf
    session['disnb']= disnb
    #cnn
    model=load_model()

    classes=['dengue', 'maleria', 'normal']
    
    frame=cv2.imread(str(pic))
    frame = cv2.resize(frame, (200,200))


    frame = cv2.cvtColor( frame, cv2.COLOR_RGB2GRAY)
    frame=frame.reshape((1,)+frame.shape)
    frame=frame.reshape(frame.shape+(1,))
    test_datagen = ImageDataGenerator(rescale=1./255)
    m=test_datagen.flow(frame,batch_size=1)
    y_pred=model.predict_generator(m,1)
    histarray2={'dengue': y_pred[0][0], 'maleria': y_pred[0][1], 'normal': y_pred[0][2]}
    update(histarray2)
    print(classes[list(y_pred[0]).index(y_pred[0].max())])
    pred=classes[list(y_pred[0]).index(y_pred[0].max())]

    session['img_pred']= pred

    
    return render_template('Results.html')


@app.route('/changeUserPass')
def changeUserPass():
    return render_template('changeUserPass.html')


@app.route('/UserHome')
def user_home():
    OPTIONS = sorted(l1)
    return render_template('UserHome.html', data=OPTIONS)


@app.route('/reg', methods=['POST'])
def register_user():
    query = "INSERT INTO user (username, password, email, mob) VALUES (:name, :pass, :email_id, :mob)"
    data = {
        'name': request.form['username'],
        'email_id': request.form['email'],
        'mob': request.form['mob'],
        'pass': request.form['password']
    }
   
    mysql.query_db(query, data)

    return render_template('UserLogin.html')


@app.route('/ulogin', methods=['POST'])
def ulogin():
    OPTIONS = sorted(l1)    
    return render_template('UserHome.html', data=OPTIONS)
    '''username = request.form['username']
    //input_password = request.form['password']
    //email_query = "SELECT * FROM user WHERE username = :uname and password = :pass"
    query_data = {'uname': str(username), 'pass': str(input_password)}
    stored_email = mysql.query_db(email_query, query_data)'''
	 
    '''if not stored_email:
        return redirect('/')

    else:
        if request.form['password'] == stored_email[0]['password']:
            OPTIONS = sorted(l1)
            return render_template('UserHome.html', data=OPTIONS)

        else:
            return redirect('/')
	'''


if __name__ == "__main__":
    app.run(debug=True)

