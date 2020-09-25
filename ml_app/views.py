from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn import linear_model
from sklearn.svm import SVC
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from csv import writer
from sklearn.neighbors import KNeighborsClassifier
from django.shortcuts import render
from . models import Enquiry

def index(request):
    flag = 0
    if request.method == "POST":
        name = request.POST["name"] 
        email = request.POST["email"]
        message = request.POST["message"]
        e = Enquiry(name=name, email=email, message=message)
        e.save()
        flag = 1
    return render(request, "ml_app/index.html", {'flag':flag})

def loan_approval(request):
    y_hat = np.empty(1)
    y_hat = [-1]
    if request.method == "POST":
        gender = request.POST["gender"]
        marital_status = request.POST["marital_status"]
        dependents = request.POST["dependents"]
        education = request.POST["education"]
        employed = request.POST["employed"]
        income = request.POST["income"]
        coincome = request.POST["coincome"]
        loan_amount = request.POST["loan_amount"]
        credit_history = request.POST["credit_history"]
        property_area = request.POST["property"]
        # ML :
        data = pd.read_csv("ml_app/static/ml_app/csv/loan_train.csv")
        df = data.replace(to_replace=["Yes", "No", "Male", "Female", "Graduate", "Not Graduate", "Urban", "Rural", "Semiurban", "3+", "N", "Y"], value=[1, 0, 1, 0, 1, 0, 2, 0, 1, 3, 0, 1])
        df = df.drop(['Loan_ID', 'Loan_Amount_Term'], 1)
        cdf = df.dropna()
        x = pd.DataFrame(cdf.iloc[:, :-1])
        y = pd.DataFrame(cdf.iloc[:, -1])
        y = y.astype('float')
        x = x.astype('float')
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=3)
        k = 3
        neigh = KNeighborsClassifier(n_neighbors=k).fit(x_train, np.ravel(y_train, order='C'))
        y_hat = neigh.predict([[gender, marital_status, dependents, education, employed, income, coincome, loan_amount, credit_history, property_area]])
        # row_content = []
        # row_content.append("LP00000")
        # row_content.append(gender)
        # row_content.append(marital_status)
        # row_content.append(dependents)
        # row_content.append(education)
        # row_content.append(employed)
        # row_content.append(income)
        # row_content.append(coincome)
        # row_content.append(loan_amount)
        # row_content.append(0)
        # row_content.append(credit_history)
        # row_content.append(property_area)
        # row_content.append(y_hat[0])
        # with open("e:/ml_project/ml_project/ml_app/loan_train.csv", 'a+', newline='') as write_obj:
        #     csv_writer = writer(write_obj)
        #     csv_writer.writerow(row_content)
    return render(request, "ml_app/loan_approval.html", {'y_hat':y_hat[0]})

def titanic_prediction(request):
    y_hat = np.empty(1)
    y_hat = [-1]
    if request.method == "POST":
        gender = request.POST["gender"]
        pclass = request.POST["pclass"]
        age = request.POST["age"]
        data = pd.read_csv("ml_app/static/ml_app/csv/titanic.csv")
        df = data[['Age', 'Sex', 'Pclass', '2urvived']]
        x = df[['Age', 'Sex', 'Pclass']]
        y = df[['2urvived']]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=5)
        svm_model = SVC(gamma = 'auto')
        svm_model.fit(x_train, np.ravel(y_train, order='C'))
        y_hat = svm_model.predict([[age, gender, pclass]])
        # row_content = []
        # row_content.append(500)
        # row_content.append(age) #age
        # row_content.append(77.24)
        # row_content.append(gender) #gender
        # row_content.append(0)
        # row_content.append(0)
        # row_content.append(0)
        # row_content.append(0)
        # row_content.append(0)
        # row_content.append(0)
        # row_content.append(0)
        # row_content.append(0)
        # row_content.append(0)
        # row_content.append(0)
        # row_content.append(0)
        # row_content.append(0)
        # row_content.append(0)
        # row_content.append(0)
        # row_content.append(0)
        # row_content.append(0)
        # row_content.append(0)
        # row_content.append(pclass) #plcass
        # row_content.append(0)
        # row_content.append(0)
        # row_content.append(0)
        # row_content.append(0)
        # row_content.append(0)
        # row_content.append(y_hat[0])
        # with open("e:/ml_project/ml_project/ml_app/titanic.csv", 'a+', newline='') as write_obj:
        #     csv_writer = writer(write_obj)
        #     csv_writer.writerow(row_content)
    return render(request, "ml_app/titanic_prediction.html", {'y_hat':y_hat[0]})

def salary_prediction(request):
    y_hat = np.empty(1)
    # y_hat = [-1]
    flag = 0
    if request.method == "POST":
        experience = request.POST["experience"]
        data = pd.read_csv("ml_app/static/ml_app/csv/Salary_Data.csv")
        x = data['YearsExperience'].values
        y = data['Salary'].values
        x = x.reshape(len(x), 1)
        y = y.reshape(len(y), 1)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
        regr = linear_model.LinearRegression()
        regr.fit(x_train, y_train)
        y_hat = regr.predict([[int(experience)]])
        flag = 1
        # row_content = []
        # row_content.append(experience)
        # row_content.append(y_hat[0])
        # with open("e:/ml_project/ml_project/ml_app/Salary_Data.csv", 'a+', newline='') as write_obj:
        #     csv_writer = writer(write_obj)
        #     csv_writer.writerow(row_content)
    return render(request, "ml_app/salary_prediction.html", {'y_hat':y_hat[0], 'flag':flag})

def diabetes_prediction(request):
    y_hat = np.empty(1)
    y_hat = [-1]
    if request.method == "POST":
        glucose = request.POST["glucose"]
        blood_pressure = request.POST["blood_pressure"]
        insulin = request.POST["insulin"]
        bmi = request.POST["bmi"]
        age = request.POST["age"]
        data = pd.read_csv("ml_app/static/ml_app/csv/diabetes.csv")        
        x = data[['Glucose', 'BloodPressure', 'Insulin', 'BMI', 'Age']].values
        y = data[['Outcome']]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)
        tree = DecisionTreeClassifier(criterion='entropy', max_depth=4)
        tree.fit(x_train, y_train)
        y_hat = tree.predict([[glucose, blood_pressure, insulin, bmi, age]])
        # row_content = []
        # row_content.append(0)
        # row_content.append(glucose)
        # row_content.append(blood_pressure)
        # row_content.append(0)
        # row_content.append(insulin)
        # row_content.append(bmi)
        # row_content.append(0)
        # row_content.append(age)
        # row_content.append(y_hat[0])
        # with open("e:/ml_project/ml_project/ml_app/diabetes.csv", 'a+', newline='') as write_obj:
        #     csv_writer = writer(write_obj)
        #     csv_writer.writerow(row_content)
    return render(request, "ml_app/diabetes_prediction.html", {'y_hat': y_hat[0]})

def student_performance(request):
    flag = 0
    y_hat = np.empty(1)
    y_hat = [-1]
    result = -1
    if request.method == "POST":
        gender = request.POST['gender']
        age = request.POST['age']
        address = request.POST['address']
        family_size = request.POST['family_size']
        parent_status = request.POST['parent_status']
        mother_edu = request.POST['mother_edu']
        father_edu = request.POST['father_edu']
        guardian = request.POST['guardian']
        travel_time = request.POST['travel_time']
        study_time = request.POST['study_time']
        failures = request.POST['failures']
        schoolsup = request.POST['schoolsup']
        famsup = request.POST["famsup"]
        paid = request.POST['paid']
        extra_curri = request.POST['extra_curri']
        nursery = request.POST['nursery']
        internet = request.POST['internet']
        family_rel = request.POST['family_rel']
        goout = request.POST['goout']
        health = request.POST['health']

        data = pd.read_csv("ml_app/static/ml_app/csv/students.csv")
        df = data[['sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'guardian', 'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'internet', 'famrel', 'goout', 'health', 'G3',]]
        df = df.replace(to_replace=["F", "M", "U", "R", "LE3", "GT3", "A", "T", "mother", "father", "other", "yes", "no"], value=[0, 1, 1, 0, 1, 0, 0, 1, 1, 2, 0, 1, 0])
        x = df[['sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'guardian', 'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'internet', 'famrel', 'goout', 'health']]
        y = df[['G3']] 
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=3)
        svm_model = SVC(gamma = 'auto')
        svm_model.fit(train_x, np.ravel(train_y, order='C'))
        y_hat = svm_model.predict([[gender, age, address, family_size, parent_status, mother_edu, father_edu, guardian, travel_time, study_time, failures, schoolsup, famsup, paid, extra_curri, nursery, internet, family_rel, goout, health]])
        flag = 1
        result =  (y_hat[0]*100)/20
    return render(request, "ml_app/student_performance.html", {'flag': flag, "y_hat":result})

def blog1(request):
    return render(request, "ml_app/blog1.html")

def blog2(request):
    return render(request, "ml_app/blog2.html")

def blog3(request):
    return render(request, "ml_app/blog3.html")

def corona_prediction(request):
    y_hat = np.empty(1)
    # y_hat = [-1]
    if request.method == "POST":
        fever = request.POST["fever"]
        tired = request.POST["tired"]
        cough = request.POST["cough"]
        diff_breadthing = request.POST["diff_breathing"]
        sore_throat = request.POST["sore_throat"]
        none_symptom = 0
        if fever==0 and tired==0 and cough==0 and diff_breadthing==0 and sore_throat==0:
            none_symptom = 1
        pains = request.POST["pains"]
        nasal_congestion = request.POST["nasal_congestion"]
        runny_nose = request.POST["runny_nose"]
        diarrhea = request.POST["diarrhea"]
        age1 = 0
        age2 = 0
        age3 = 0
        age4 = 0
        age5 = 0
        age = request.POST["age"]
        if age == 1:
            age1 = 1
        elif age == 2:
            age2 = 1
        elif age == 3:
            age3 = 1
        elif age == 4:
            age4 = 1
        elif age == 5:
            age5 = 1
        male = 0
        female = 0
        other = 0
        gender = request.POST["gender"]
        if gender == 0:
            male = 1
        elif gender == 1:
            female = 1
        elif gender == 2:
            other = 1
        mild = 0
        moderate = 0
        severe = 0
        none = 0    
        severity = request.POST["severity"]
        if severity == 1:
            mild = 1
        elif severity == 2:
            moderate = 1
        elif severity == 3:
            severe = 1
        elif severity == 4:
            none = 1
        data = pd.read_csv("ml_app/static/ml_app/csv/corona.csv")
        print("data read sucess")
        df = data.drop(['Country', 'None_Experiencing', 'Contact_Dont-Know', 'Contact_No', 'Contact_Yes'], 1)
        kmeans = KMeans(n_clusters=2)
        kmeans.fit(df)
        print("model fit sucess")
        y_hat = kmeans.predict([[fever, tired, cough, diff_breadthing, sore_throat, none_symptom, pains, nasal_congestion, runny_nose, diarrhea, age1, age2, age3, age4, age5, female, male, other, mild, moderate, none, severe]])
        print("getting predictions sucessfully")
    return render(request, "ml_app/corona_prediction.html", {'yhat': y_hat[0]})

def blog4(request):
    return render(request, "ml_app/blog4.html")