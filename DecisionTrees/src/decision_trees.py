import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


class Printer(object) :

    @staticmethod
    def display(y_prediction, y_test) :
        print("\nAccuracy : ", accuracy_score(y_test, y_prediction) * 100)
        print("\nReport : \n", classification_report(y_test, y_prediction))


class Parser(Printer) :
    def __init__(self) :
        pass

    def parse_method(self) :
        try :
            data = pd.read_csv('./input/HR-Em.csv')
            data.isnull().sum()

            age = pd.get_dummies(data['Over18'], '20')
            gender = pd.get_dummies(data['Gender'], 'sex')
            education = pd.get_dummies(data['EducationField'], 'field')
            role = pd.get_dummies(data['JobRole'], 'role')
            department = pd.get_dummies(data['Department'], 'dept')
            data = pd.concat([data, age, gender, education, role, department], axis=1)

            dataFrame = data.select_dtypes(include=['int64', 'uint8'])

            X = dataFrame[['Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EmployeeCount',
                           'EmployeeNumber', 'EnvironmentSatisfaction', 'HourlyRate',
                           'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MonthlyIncome',
                           'MonthlyRate', 'NumCompaniesWorked', 'PercentSalaryHike',
                           'PerformanceRating', 'RelationshipSatisfaction', 'StandardHours',
                           'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear',
                           'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole',
                           'YearsSinceLastPromotion', 'YearsWithCurrManager', '20_Y', 'sex_Female', 'sex_Male',
                           'field_Human Resources', 'field_Life Sciences', 'field_Marketing',
                           'field_Medical', 'field_Other', 'field_Technical Degree',
                           'role_Healthcare Representative', 'role_Human Resources',
                           'role_Laboratory Technician', 'role_Manager',
                           'role_Manufacturing Director', 'role_Research Director',
                           'role_Research Scientist', 'role_Sales Executive',
                           'role_Sales Representative', 'dept_Human Resources', 'dept_Research & Development',
                           'dept_Sales']]
            y = data['JobRole']

            # Splitting the dataset into train and test
            X_train, X_test, y_train, y_test = train_test_split(X, y)

            self.using_entropy_criterion(X_test, X_train, y_test, y_train)

            self.using_gini_criterion(X_test, X_train, y_test, y_train)

        except Exception as exception :
            print(f"Filed to run the implementation due to: {exception}\n")

    # Function to perform training with entropy.
    @staticmethod
    def using_entropy_criterion(X_test, X_train, y_test, y_train) :

        # Decision tree with entropy
        classifier_entropy = DecisionTreeClassifier(criterion="entropy", random_state=100,
                                                    max_depth=3, min_samples_leaf=5)
        # Performing training
        classifier_entropy.fit(X_train, y_train)
        y_prediction_entropy = classifier_entropy.predict(X_test)
        Parser.display(y_prediction_entropy, y_test)

        tree.plot_tree(classifier_entropy,  # visualising the tree decision
                       label='all',  # every box has label we can read
                       class_names=sorted(y_train.unique()),  # display the class
                       filled=True,  # each bar is filled with a color
                       fontsize=7)
        # plt.scatter(y_prediction_entropy, y_test, alpha=.8, color='r')
        plt.show()

    # Function to perform training with giniIndex.
    @staticmethod
    def using_gini_criterion(X_test, X_train, y_test, y_train) :
        # Creating the classifier object
        classifier_gini = DecisionTreeClassifier(criterion="gini", random_state=100,
                                                 max_depth=3, min_samples_leaf=5)

        # Performing training
        classifier_gini.fit(X_train, y_train)

        # Prediction using gini
        y_prediction_gini = classifier_gini.predict(X_test)
        Parser.display(y_prediction_gini, y_test)

        tree.plot_tree(classifier_gini,  # visualising the tree decision
                       label='all',  # every box has label we can read
                       class_names=sorted(y_train.unique()),  # display the class
                       filled=True,  # each bar is filled with a color
                       fontsize=7)

        #  plt.scatter(y_prediction_gini, y_test, alpha=.8, color='r')
        plt.show()
