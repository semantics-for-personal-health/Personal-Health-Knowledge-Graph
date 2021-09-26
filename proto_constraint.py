#! python3
from proto_lib import *
import string
from saxpy.paa import paa
import json
import datetime
import numpy as np
from saxpy.hotsax import find_discords_hotsax
import time
import csv
import sys
from rdflib import Graph, Literal, RDF, URIRef, Namespace
from rdflib.namespace import PROV, RDFS, FOAF, XSD

### Input parameters ###
attributes = ["MyFitnessPal"]
attr_index = 0
age = 23
activity_level = "active"
alpha_size = 5
alpha = None
alphabet = string.ascii_lowercase
rdf_file = open("phkg.rdf","w")

phkg = Graph()
SIO = Namespace("http://semanticscience.org/ontology/sio.owl")
PATO = Namespace("http://purl.obolibrary.org/obo/pato.owl")
user = URIRef('http://idea.rpi.edu/heals/phkg/user')

# Properties
has_attribute = URIRef('http://semanticscience.org/resource/SIO_000008')
is_related_to = URIRef('http://semanticscience.org/resource/SIO_000001')
describes = URIRef('http://semanticscience.org/resource/SIO_000563')
has_value = URIRef('http://semanticscience.org/resource/SIO_000300')
refers_to = URIRef('http://semanticscience.org/resource/SIO_000628')
has_participant = URIRef('http://semanticscience.org/resource/SIO_000132')
has_quality = URIRef('http://semanticscience.org/resource/SIO_000217')
has_time_boundary = URIRef('http://semanticscience.org/resource/SIO_000679')
has_frequency = URIRef('http://semanticscience.org/resource/SIO_000900')
is_described_by = URIRef('http://semanticscience.org/resource/SIO_000557')
is_causally_related_to = URIRef('http://semanticscience.org/resource/SIO_000294')

# Classes
frequency = URIRef('http://semanticscience.org/resource/SIO_001367')
tendency = URIRef('http://purl.obolibrary.org/obo/PATO_0002360')
SelectingNewRecipes = URIRef('http://idea.rpi.edu/heals/phkg/SelectingNewRecipes')
pattern = URIRef('http://semanticscience.org/resource/SIO_000130')
Relationship = URIRef('http://purl.obolibrary.org/obo/NCIT_C25648')
Sequence = URIRef('http://purl.obolibrary.org/obo/NCIT_C25673')
SequenceLength = URIRef('http://edamontology.org/data_1249')
Food = URIRef('http://purl.obolibrary.org/obo/NCIT_C62695')
ConsistentCarbohydrateIntake = URIRef('http://idea.rpi.edu/heals/phkg/ConsistentCarbohydrateIntake')
Meal = URIRef('http://purl.obolibrary.org/obo/NCIT_C80248')
Breakfast = URIRef('http://purl.obolibrary.org/obo/NCIT_C80249')
Lunch = URIRef('http://purl.obolibrary.org/obo/NCIT_C80250')
Dinner = URIRef('http://purl.obolibrary.org/obo/NCIT_C80251')
CarbohydrateIntake = URIRef('http://www.enpadasi.eu/ontology/release/v1/ons/ONS_0000023')
Prediction = URIRef('http://purl.obolibrary.org/obo/NCIT_C54156')
Week = URIRef('http://semanticscience.org/resource/SIO_001354')
CoefficientOfVariation = URIRef('http://purl.obolibrary.org/obo/STATO_0000236')
HighCarbLowFat = URIRef('http://idea.rpi.edu/heals/phkg/HighCarbLowFat')
ProgressReport = URIRef('http://idea.rpi.edu/heals/phkg/ProgressReport')
Trend = URIRef('http://purl.allotrope.org/ontologies/result#AFR_0000634')
Slope = URIRef('http://purl.obolibrary.org/obo/NCIT_C70744')
NutrientIntakeGoal = URIRef('http://idea.rpi.edu/heals/phkg/NutrientIntakeGoal')
HighCarb = URIRef('http://idea.rpi.edu/heals/phkg/HighCarb')
LowFat = URIRef('http://idea.rpi.edu/heals/phkg/LowFat')
Weekday = URIRef('http://purl.obolibrary.org/obo/NCIT_C86936')
Sunday = URIRef('http://purl.obolibrary.org/obo/NCIT_C64968')
Monday = URIRef('http://purl.obolibrary.org/obo/NCIT_C64962')
Tuesday = URIRef('http://purl.obolibrary.org/obo/NCIT_C64963')
Wednesday = URIRef('http://purl.obolibrary.org/obo/NCIT_C64964')
Thursday = URIRef('http://purl.obolibrary.org/obo/NCIT_C64965')
Friday = URIRef('http://purl.obolibrary.org/obo/NCIT_C64966')
Saturday = URIRef('http://purl.obolibrary.org/obo/NCIT_C64967')
IntakeAnomaly = URIRef('http://idea.rpi.edu/heals/phkg/IntakeAnomaly')
Today = URIRef('http://purl.obolibrary.org/obo/NCIT_C41134')
FoodPreference = URIRef('http://purl.obolibrary.org/obo/NBO_0000141')

# Subclasses
phkg.add((Breakfast, RDFS.subClassOf, Meal))
phkg.add((Lunch, RDFS.subClassOf, Meal))
phkg.add((Dinner, RDFS.subClassOf, Meal))
phkg.add((Sunday, RDFS.subClassOf, Weekday))
phkg.add((Monday, RDFS.subClassOf, Weekday))
phkg.add((Tuesday, RDFS.subClassOf, Weekday))
phkg.add((Wednesday, RDFS.subClassOf, Weekday))
phkg.add((Thursday, RDFS.subClassOf, Weekday))
phkg.add((Friday, RDFS.subClassOf, Weekday))
phkg.add((Saturday, RDFS.subClassOf, Weekday))

date_columns = { "Weather" : "YEARMODA",
                     "Stock Market Data" : "date",
                     "Step Count" : "date",
                     "Heart Rate" : "date",
                     "ActivFit" : "date",
                     "Calorie Intake" : "date",
                     "MyFitnessPal" : "date",
                     "MyFitnessPalMeals" : "date",
                     "StepUp" : "date",
                     "Cue" : "date",
                     "Energy Deficit": "date",
                     "SatFatDecrease": "date",
                     "FoodPreferences" : "date"
                     }

weekday_dict = {1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday', 5: 'Friday', 6: 'Saturday', 7: 'Sunday'}

key_dict = {"MyFitnessPal" : ["Protein","Carbohydrate Intake","Fat Intake"],
            "MyFitnessPalMeals" : ["Breakfast","Lunch","Dinner","Total"],
            "Energy Deficit" : ["Energy Deficit"],
            "SatFatDecrease": ["Saturated Fat"]
            }

df_key_dict = {"Calorie Intake" : "Calories",
               "Carbohydrate Intake" : "Carbohydrates",
               "Carb Intake" : "Carbohydrates",
               "Energy Deficit" : "Energy Deficit",
               "Saturated Fat" : "Saturated Fat",
               "Fat Intake" : "Fat",
               "Protein" : "Protein",
               "Sodium" : "Sodium",
               "Sugar" : "Sugar"}

def get_data(attr,df_index_list):
    df_lists, pid_list = get_data_list(df_index_list,attr,constraints=True)
    return df_lists

def build_sax_list(key_list,data,goals,alphabet,alpha_size):
    summarizer_type = "Past Daily TW"
    letter_map = None
    summarizers = ["reach","did not reach"]

    sax_list = []
    tw_sax_list = []
    binary_list = []
    for i in range(len(key_list)):
        sax = ""
        attr = key_list[i]
        time_series = data[i]
        summarizer = summarizers[i]
        goals_ = goals[i]
        for j in range(len(time_series)):
            value = time_series[j]
            goal = goals_[i]
            muS = int(get_muS(attr,summarizer_type,summarizer,value,letter_map,alpha_size,goal_=goal))
            sax += alphabet[muS]

        tw = 7
        tw_sax = []
        for j in range(0,len(sax),tw):
            subsax = sax[j:j+tw]
            tw_sax.append(subsax)

        sax_list.append(sax)
        tw_sax_list.append(subsax)

    for i in range(len(sax_list[0])):
        cnt = 0
        for j in range(len(sax_list)):
            if sax_list[j][i] == 'b':
                cnt += 1

        binary_list.append(int(cnt == len(sax_list)))

    return sax_list, tw_sax_list, binary_list

def build_time_series_daily(food_set,food_data):
    series_dict = dict()
    for food in food_set:
        food_series = ''

        for day in food_data:
            food_series += str(int(food in day))
        series_dict[food] = food_series

    return series_dict

def build_time_series_meal(food_set,food_data):
    series_dict = dict()
    for food in food_set:
        food_series = ''

        for day in food_data:
            for meal in day:
                food_series += str(int(food == meal))
        series_dict[food] = food_series

    return series_dict

def find_preference(food,data):
    r = round(float(data.count('1'))/len(data),2)
    #rdf_file.write("< " + food + " | hasEatingFrequency | " + str(r) + " >")
    return [food,r]

def find_behavior_sequences(data):

    tw = len(data)
    db_fn_prefix = "series_db"

    path = "" # Path for pattern data
    cygwin_path = r"" # Path to Cygwin
    min_conf = 0
    min_sup = 0
    proto_cnt = 0
    alpha_sizes = [2]

    letter_map_list = [{'a' : 0,
                        'b' : 1}]

    alphabet_list = ['ab']

    attr_list = ["high-carb, low-fat diet"]

    sax = []
    for data_point in data:
        letter = "b" if data_point else "a"
        sax.append(letter)

    patterns = analyze_patterns(attr_list,[sax],alphabet_list,letter_map_list,weekday_dict,tw,alpha_sizes,db_fn_prefix,path,cygwin_path,min_conf,min_sup,proto_cnt,food_items=True)

    if len(patterns) > 0:
        for p in patterns:

            tr = URIRef('http://purl.obolibrary.org/obo/NCIT_C25648tr')
            dp1 = URIRef('http://semanticscience.org/resource/SIO_000130/dp1')
            dp2 = URIRef('http://semanticscience.org/resource/SIO_000130/dp2')
            ir = URIRef('http://purl.obolibrary.org/obo/NCIT_C25648ir')
            s1 = URIRef('http://semanticscience.org/resource/SIO_001367/s1')

            phkg.add((tr, RDF.type, Relationship))
            phkg.add((user, has_attribute, tr))
            phkg.add((dp1, RDF.type, pattern))
            phkg.add((dp2, RDF.type, pattern))
            phkg.add((tr, has_attribute, dp1))
            phkg.add((tr, has_attribute, dp2))
            phkg.add((dp1, is_related_to, dp2))
            phkg.add((ir, RDF.type, Relationship))
            phkg.add((tr, describes, ir))
            phkg.add((s1, RDF.type, frequency))
            phkg.add((tr, has_frequency, s1))
            phkg.add((s1, has_value, Literal(p[2],datatype=XSD.float)))

def find_food_sequences(food,data):

    tw = len(data)
    db_fn_prefix = "series_db"

    path = "" # Path for pattern data
    cygwin_path = r"" # Path to Cygwin
    min_conf = 0.8
    min_sup = 0.2
    proto_cnt = 0
    alpha_sizes = [2,2]

    sax = ''
    for data_point in data:
        if int(data_point):
            sax += 'b'
        else:
            sax += 'a'

    letter_map_list = [{'a' : 0,
                        'b' : 1}]

    alphabet_list = ['ab']
    patterns = analyze_patterns([food],[sax],alphabet_list,letter_map_list,weekday_dict,tw,alpha_sizes,db_fn_prefix,path,cygwin_path,min_conf,min_sup,proto_cnt,food_items=True)

    max_cnt = 0
    for pattern in patterns:
        full_pattern = pattern[0] + pattern[1]
        cnt = full_pattern.count('1')

        if cnt > max_cnt:
            max_cnt = cnt
            conf = pattern[2]
            supp = pattern[3]

    if max_cnt > 0:


        ds = URIRef('http://purl.obolibrary.org/obo/NCIT_C25673/ds')
        sl = URIRef('http://edamontology.org/data_1249/sl')
        f4 = URIRef('http://purl.obolibrary.org/obo/NCIT_C62695/f4')
        s2 = URIRef('http://semanticscience.org/resource/SIO_001367/s2')

        phkg.add((ds, RDF.type, Sequence))
        phkg.add((user, has_attribute, ds))
        phkg.add((sl, RDF.type, SequenceLength))
        phkg.add((ds, has_attribute, sl))
        phkg.add((sl, has_value, Literal(max_cnt,datatype=XSD.int)))
        phkg.add((f4, RDF.type, Food))
        phkg.add((ds, describes, f4))
        phkg.add((f4, RDFS.label, Literal(food)))
        phkg.add((s2, RDF.type, frequency))
        phkg.add((ds, has_attribute, s2))
        phkg.add((s2, has_value, Literal(supp,datatype=XSD.float)))

    return

def find_relationship(attr,food1,food2,series_dict_f,letter_map_list,alpha,alpha_sizes,age,activity_level,TW):

    summarizer_list = [None,None]

    start_day = 0
    end_day = len(series_dict_f[food1])-1

    key_list = [food1,food2]
    past_tw_list = [series_dict_f[food1],series_dict_f[food2]]

    cnt = 0
    quotient = 0
    cnt_ = 0
    quotient_ = 0
    for i in range(len(series_dict_f[food1])):
        if series_dict_f[food2][i]=='1': # When Food2 is eaten, Food1 is eaten
            quotient_ += 1
            if series_dict_f[food1][i]=='1':
                cnt_ += 1

        if series_dict_f[food1][i]=='1':
            quotient += 1
            if series_dict_f[food2][i]=='1':
                cnt += 1

    freq_ = round(float(cnt)/quotient,2) # not a mistake
    freq = round(float(cnt_)/quotient_,2)

    return freq, freq_

def get_culprits(attr,data,summarizer_type,letter_map_list,alpha_sizes,TW,goals,meal_keys,guideline):
    culprit_summaries = []
    culprit_frequencies = []
    culprit_summarizers = ["too low","too high"]
    meal_keys = list(meal_keys)
    max_frequency = 0
    culprit_summary = "To help maintain a relatively fixed carbohydrate intake, consider"
    sub_summary = ""
    for i in range(len(meal_keys)):

        key = meal_keys[i]
        if key == "Total" or key == "Snacks" or key == "date":
            continue


        carb_list = []
        for j in range(len(data[key])):
            try:
                carb_list.append(data[key][j]["Carbohydrates"])
            except TypeError:
                carb_list.append(data[key][j])

        avg_list, t1_list, quantifier_list, summary_list, summarizer_list = generate_summaries([culprit_summarizers],summarizer_type,[attr],[carb_list],[letter_map_list[i]],[alpha_sizes[i]],alpha,age=age,activity_level=activity_level,TW=TW,goals=[goals[i]],ada_goal="culprit")

        if quantifier_list != None:
            index = best_quantifier_index(quantifier_list,t1_list)
            quantifier = quantifier_list[index]
            frequency = avg_list[index]
            summary = summary_list[index]

            if "none" in quantifier:
                continue

            culprit_frequencies.append(frequency)
            culprit_summaries.append(summary)

            if frequency > max_frequency:
                sub_summary = summary.split(", consider")[-1]
                max_frequency = frequency

    culprit_summary += sub_summary

    change_set = ["increasing","decreasing"]
    rec = None
    for change in change_set:
        if change in culprit_summary:
            rec = change
    rec = rec.capitalize() + "Consumption"

    meal = None
    for meal_ in meal_keys:
        if meal_.lower() in culprit_summary:
            meal = meal_
    meal = meal.capitalize()

    su = URIRef('http://idea.rpi.edu/heals/phkg/Suggestion/su')
    cci = URIRef('http://idea.rpi.edu/heals/phkg/ConsistentCarbohydrateIntake/cci')

    if meal == "Breakfast":
        me = URIRef('http://purl.obolibrary.org/obo/NCIT_C80249/me')
        phkg.add((me, RDF.type, Breakfast))
    elif meal == "Lunch":
        me = URIRef('http://purl.obolibrary.org/obo/NCIT_C80250/me')
        phkg.add((me, RDF.type, Lunch))
    elif meal == "DinnerMeal":
        me = URIRef('http://purl.obolibrary.org/obo/NCIT_C80251/me')
        phkg.add((me, RDF.type, Dinner))

    ci = URIRef('http://www.enpadasi.eu/ontology/release/v1/ons/ONS_0000023/ci')

    phkg.add((su, RDF.type, Suggestion))
    phkg.add((cci, RDF.type, ConsistentCarbohydrateIntake))
    phkg.add((su, is_related_to, cci))
    phkg.add((su, refers_to, me))
    phkg.add((ci, RDF.type, CarbohydrateIntake))
    phkg.add((su, refers_to, ci))
    phkg.add((su, has_attribute, i))

    return culprit_summary

def membership_function(data,TW,full=False):
    avg = sum(data)/float(len(data))
    if TW == "weeks" and len(data) == 21:
        full = True

    deviation = 0
    for value in data:
        deviation += (value - avg)**2

    deviation = math.sqrt(deviation/(len(data)-1))

    if avg:
        cv = deviation/avg
    else:
        cv = 0

    diff = max(1-cv,0)
    return diff, full, avg

def relative_fix(attr,df_list,TW,guideline,letter_map,weekday=None,ada_goal="consistentcarb"):

    D = []
    tw_data = []
    day_data = []
    L = None

    for i in range(len(df_list["date"])):
        sub_day_dict = dict()
        if (TW == "weeks" and (i%7==0 or i==0)) or TW == "days" or (TW == None and i==0):
            new_dict = True
            sub_tw_dict = dict()
        else:
            new_dict = False

        for key in list(df_list.keys()):
            if key == "date":
                continue
            if key in sub_tw_dict:
                sub_tw_dict[key].append(df_list[key][i]["Carbohydrates"])
            else:
                sub_tw_dict[key] = [df_list[key][i]["Carbohydrates"]]

            sub_day_dict[key] = df_list[key][i]["Carbohydrates"]

        if new_dict:
            tw_data.append(sub_tw_dict)
        day_data.append(sub_day_dict)

    if len(df_list["date"])%7 != 0:
        tw_data.append(sub_tw_dict)
    if len(df_list["date"]) >= len(tw_data[-1])+14:
        last_tw_len = len(tw_data[-1])+14
    else:
        last_tw_len = len(tw_data[-1])

    # Find percent difference of standard deviation
    tw_diffs = []
    avg_list = []
    for time_window in tw_data:
        sub_data = []
        for key in time_window.keys():
            if key == "Total" or key == "Snacks":
                continue

            for i in range(len(time_window[key])): # All meals in TW
                sub_data.append(time_window[key][i])

        tw_diff, full, avg = membership_function(sub_data,TW)
        tw_diffs.append(tw_diff)
        avg_list.append(avg)

    scores = []
    if weekday != None:
        for i in range(len(tw_data[0]["Breakfast"])):
            sublist = []
            for key in tw_data[0].keys():
                sublist.append(tw_data[0][key][i])
            score,_ = membership_function(sublist,TW)
            scores.append(score)

    day_diffs = []
    for day in day_data:
        sub_data = []
        for key in day.keys():
            if key == "Total" or key == "Snacks":
                continue

            sub_data.append(day[key])

        day_diff, full, avg = membership_function(sub_data,TW)
        day_diffs.append(day_diff)

    index = -1
    if len(tw_data[-1]["Breakfast"]) < 7:
        index = -2
    try:
        value = tw_diffs[index:][0]
        avg = round(avg_list[index:][0],1)
    except IndexError:
        return None

    #rdf_file.write(":p a prov:Person .\n")
    #rdf_file.write(":p sio:has-attribute :cci .\n")
    #rdf_file.write(":cci a ConsistentCarbohydrateIntake .\n")

    summarizer_type = "ADA Goal Evaluation - MyFitnessPalMeals"
    letter_map_list = [letter_map]*len(tw_data[index].keys())
    guideline_summarizers = ["reached"]
    alpha_sizes = [alpha_size]*len(tw_data[index].keys())
    summarizers_list = [guideline_summarizers]*len(tw_data[index].keys())
    goals = []
    for key in tw_data[index].keys():
        goal = ["consistentcarb",key]
        goals.append([goal]*len(tw_data[index][key]))

    #culprit_summary = get_culprits(attr,tw_data[index],summarizer_type,letter_map_list,alpha_sizes,TW,goals,tw_data[index].keys(),guideline)

    quantifiers = ["not","slightly","moderately","considerably","definitely"]
    relative_summarizer, truth = getQForS(value,None,None,q_list=quantifiers)

    relative_summary = "This past full week, "
    keep_tense = " keeping"

    if relative_summarizer == "not":
        relative_summarizer += " done"
    else:
        relative_summarizer = "done " + relative_summarizer

    relative_summary += "you have " + relative_summarizer + " well at"+ keep_tense + " your carbohydrate intake relatively fixed"

    cci = URIRef('http://idea.rpi.edu/heals/phkg/ConsistentCarbohydrateIntake/cci')
    cv = URIRef('http://purl.obolibrary.org/obo/STATO_0000236/cv')

    phkg.add((cv, RDF.type, CoefficientOfVariation))
    phkg.add((cci, RDF.type, ConsistentCarbohydrateIntake))
    phkg.add((cci, has_attribute, cv))
    phkg.add((cv, has_value, Literal(avg,datatype=XSD.float)))

    return relative_summary

def general_ADA_summary(df_list,attr,ada_goal,tw,guideline,TW,singular_TW,letter_map):
    """
    Inputs:
    - df_list: the relevant data
    - attr: the attribute of interest
    - persona_list: the most recent version of the list of information pertaining to a persona
    - ada_goal: the ADA guideline goal
    - TW: the time window in words (e.g. "weeks")
    - guideline: the ADA guideline

    - tw: the time window in numeric form (e.g. a week is 7)
    - persona: the persona ID
    - singular_TW: the singular form of TW (e.g. "week")

    """

    sax_list = []
    tw_sax_list = []
    data_list = []
    summary_data_list = []
    past_full_wks = []

    if type(df_list) is list:
        df_list = df_list[0]

    date_column_list = [df_list[key] for key in df_list.keys() if key == date_columns[attr]]

    date_column = date_column_list[0].copy()

    for i in range(len(date_column)):
        date = date_column[i]

        # Parse dates
        if attr == "Energy Deficit" or attr == "SatFatDecrease":
            date = date.split('/')
            date = datetime.datetime(int(date[2]),int(date[0]),int(date[1]))
        elif attr == "MyFitnessPal" or attr == "MyFitnessPalMeals" or attr == "FoodPreferences":
            if type(date) == float:
                continue
            date = date.split('/')
            date = datetime.datetime(int(date[2]),int(date[0]),int(date[1]))

        week_day = date.weekday()
        week_day = weekday_dict[week_day+1]
        date_column[i] = week_day

    attr_list = key_dict[attr]
    summary = None

    from itertools import combinations
    # Try different combinations of attributes
    combos = []
    for i in range(len(attr_list)):
        comb = combinations(attr_list,i+1)
        combos.append(list(comb))

    if attr == "MyFitnessPalMeals":
        combos = [combos[-1]]

    for combo in combos:

        for key_list in combo:
            key_list = list(key_list)

            if ada_goal == "highcarblowfat" and key_list != ["Carbohydrate Intake","Fat Intake"]:
                continue

            alphabet_list = []
            letter_map_list = []
            alpha_sizes = []
            sax_list = []
            tw_sax_list = []
            goals = None
            if attr == "MyFitnessPal":
                goals = []
                for key in key_list:
                    tmp = []
                    for i in range(len(df_list["Total"])):
                        if ada_goal == "fat percentage":
                            tmp.append([ada_goal,df_list["Total"][i]["Calories"],df_list["Total"][i]["Fat"]])
                        elif ada_goal == "highcarblowfat":
                            tmp.append([ada_goal,df_list["Total"][i]["Calories"],df_list["Total"][i]["Carbohydrates"],df_list["Total"][i]["Fat"]])
                        elif ada_goal == "lowcarb":
                            tmp.append([ada_goal,df_list["Total"][i]["Calories"],df_list["Total"][i]["Carbohydrates"]])
                        else:
                            tmp.append(None)
                    goals.append(tmp)
            elif attr == "Energy Deficit":
                tmp = []
                for i in range(len(df_list["Total"])):
                    tmp.append([ada_goal])
                goals = tmp

            summary_data_list = []
            error = False
            for i in range(len(key_list)):
                data = [x[df_key_dict[key_list[i]]] for x in df_list["Total"]]
                if len(data) < tw:
                    error = True
                    break

                data_list.append(data)
                alphabet_list.append(alphabet)
                letter_map_list.append(letter_map)
                alpha_sizes.append(alpha_size)

                full_sax_rep = ts_to_string(znorm(np.array(data)), cuts_for_asize(alpha_sizes[i]))

                if tw > 0:
                    tw_sax = ts_to_string(paa(znorm(np.array(data)),int(len(data)/tw)), cuts_for_asize(alpha_sizes[i]))
                    tw_sax_list.append(tw_sax)

                    prev_start_day = int(tw*(len(tw_sax)-2))
                    start_day = int(tw*(len(tw_sax)-1))
                    end_day = int(tw*len(tw_sax))
                    other_start_day = int(tw*(len(tw_sax)/2))
                    other_end_day = int(tw*(len(tw_sax)/2)+tw)

                    past_full_wks.append(data[start_day:end_day])

                    other_tw = data[other_start_day:other_end_day]
                    other_day_sax = full_sax_rep[other_start_day:other_end_day]
                    other_days = date_column[other_start_day:other_end_day]
                else:
                    start_day = 0
                    end_day = len(full_sax_rep)

                summary_data = full_sax_rep[start_day:end_day]
                x_vals = "days"

                sax_list.append(full_sax_rep)
                summary_data_list.append(summary_data)

            if error:
                continue

            sub_dict = dict()
            if ada_goal == "monthly summary":
                if len(key_list)!=1:
                    continue

                calorie_avg = sum(df_list["Calories"][-30:])/30
                carb_avg = sum(df_list["Carbohydrates"][-30:])/30
                fat_avg = sum(df_list["Fat"][-30:])/30
                protein_avg = sum(df_list["Protein"][-30:])/30

                carb_percentage = ((carb_avg*4)/calorie_avg)*100
                fat_percentage = ((fat_avg*9)/calorie_avg)*100
                protein_percentage = ((protein_avg*4)/calorie_avg)*100

                if carb_percentage < 45 and "DG17" not in persona_dict[persona]["guidelines"]:
                    carb_summarizer = "below"
                elif carb_percentage < 45 and "DG17" in persona_dict[persona]["guidelines"]:
                    carb_summarizer = "within"
                elif carb_percentage >= 45 and carb_percentage <= 65:
                    carb_summarizer = "within"
                else:
                    carb_summarizer = "above"

                if fat_percentage < 10:
                    fat_summarizer = "below"
                elif fat_percentage >= 10 and fat_percentage <= 35:
                    fat_summarizer = "within"
                else:
                    fat_summarizer = "above"

                if protein_percentage < 20:
                    protein_summarizer = "below"
                elif protein_percentage >= 20 and protein_percentage <= 35:
                    protein_summarizer = "within"
                else:
                    protein_summarizer = "above"

                carb_summary = "In the past full month, your average carbohydrate intake was " + carb_summarizer + " the desired range."
                fat_summary = "In the past full month, your average fat intake was " + fat_summarizer + " the desired range."
                protein_summary = "In the past full month, your average protein intake was " + protein_summarizer + " the desired range."

                if "DG17" in persona_dict[persona]["guidelines"]:
                    rangeMax = calorie_avg*0.45
                    rangeMin = 0.0
                else:
                    rangeMax = calorie_avg*0.65
                    rangeMin = calorie_avg*0.45

            else:
                guideline_summarizers = ["reached","did not reach"]
                if attr == "StepUp" or ada_goal != None:
                    guideline_summarizers = ["reached"]
                summarizers_list = []
                for i in range(len(key_list)):
                    summarizers_list.append(guideline_summarizers)
                if attr == "Activity":
                    past_tw = []
                    for letter in summary_data:
                        past_tw.append(categ_eval(letter))

                summarizer_type = "ADA Goal Evaluation - "
                for i in range(len(key_list)):
                    if key_list[i] == "step count":
                        key_list[i] = "Step Count"

                    summarizer_type += key_list[i]
                    if i != len(key_list)-1:
                        summarizer_type += " and "

                past_tw_list = []
                for i in range(len(key_list)):
                    past_tw_list.append(data_list[i][start_day:end_day])

                input_goals = [goals]
                if attr == "MyFitnessPal" or attr == "MyFitnessPalMeals":
                    input_goals = goals

                range_dict = dict()
                #input(past_tw_list[0])
                avg_list, t1_list, quantifier_list, summary_list, summarizer_list, _ = generate_summaries(summarizers_list,summarizer_type,key_list,past_tw_list,letter_map_list,alpha_sizes,alpha,age=age,activity_level=activity_level,TW=TW,goals=input_goals,ada_goal=ada_goal,range_dict=range_dict)

                if quantifier_list != None:
                    index = best_quantifier_index(quantifier_list,t1_list)
                    goal_summary = summary_list[index]
                    summarizers = summarizer_list[index]
                    truth = t1_list[index]
                    average = avg_list[index]
                else:
                    goal_summary = ""

                if len(goal_summary) != 0:
                    if goal_summary not in sub_dict.keys():
                        sub_dict[goal_summary] = dict()
                    if attr == "StepUp":
                        treatment = df_list["Treatment"][0]
                        if treatment not in sub_dict[goal_summary].keys():
                            sub_dict[goal_summary][treatment] = 1
                        else:
                            sub_dict[goal_summary][treatment] += 1

                    if (attr == "Energy Deficit" and persona) or attr != "Energy Deficit":
                        goal_summary = goal_summary.replace("reached your goal to obtain","obtained")


    cf = URIRef('http://idea.rpi.edu/heals/phkg/HighCarbLowFat/cf')
    fr4 = URIRef('http://semanticscience.org/resource/SIO_001367/fr4')

    phkg.add((cf, RDF.type, HighCarbLowFat))
    phkg.add((fr4, RDF.type, frequency))
    phkg.add((user, has_attribute, cf))
    phkg.add((cf, has_attribute, fr4))
    phkg.add((fr4, has_value, Literal(average,datatype=XSD.float)))

    return

def long_term_trend(data):
    import scipy.stats as sp
    import numpy as np

    tw = 7
    avgs = []
    for i in range(len(data)):
        subset = data[i:i+tw]
        avg = float(sum(subset))/len(subset)
        avgs.append(avg)

    y = np.array(data, dtype=float)
    x = np.array([x for x in range(len(avgs))],dtype=float)
    slope, _, _, _, _ = sp.linregress(x,y)

    cci = URIRef('http://idea.rpi.edu/heals/phkg/ConsistentCarbohydrateIntake/cci')
    tr1 = URIRef('http://purl.allotrope.org/ontologies/result#AFR_0000634/tr1')
    sl1 = URIRef('http://purl.obolibrary.org/obo/NCIT_C70744/sl1')

    phkg.add((cci, RDF.type, ConsistentCarbohydrateIntake))
    phkg.add((tr1, RDF.type, Trend))
    phkg.add((sl1, RDF.type, Slope))

    phkg.add((user, has_attribute, cci))
    phkg.add((user, has_attribute, tr1))
    phkg.add((tr1, is_described_by, sl1))
    phkg.add((tr1, refers_to, cci))
    phkg.add((sl1, has_value, Literal(round(slope,2),datatype=XSD.float)))

def main():

    # Bind namespaces
    phkg.bind("sio",SIO)
    phkg.bind("foaf",FOAF)
    phkg.bind("rdf",RDF)
    phkg.bind("prov",PROV)
    phkg.bind("rdfs",RDFS)

    phkg.add((user, RDF.type, PROV.Person))

    alpha_sizes = [None,None]
    letter_map_list = [None,None]
    alphabet_list = [None,None]
    age = 24
    activity_level = "sedentary"
    TW = None

    # Gather data
    attr = attributes[attr_index]
    df_index_list = [0]
    data = get_data(attr,df_index_list)

    meal_column = data[0]["Meal"]
    date_column = data[0]["date"]
    meal_set = set(meal_column)
    meal_set.add("Total")
    food_data = [x for x in data[0]["Food"] if type(x) is str]
    ingredient_data = [x for x in data[0]["Ingredients"] if type(x) is str]

    calorie_dict = dict()
    calorie_data = data[0]["Calories"]
    for i in range(len(calorie_data)):
        date = date_column[i]
        if date not in calorie_dict.keys():
            calorie_dict[date] = calorie_data[i]
        else:
            calorie_dict[date] += calorie_data[i]

    calorie_data = [calorie_dict[key] for key in sorted(calorie_dict.keys())]

    carb_dict = dict()
    carb_data = data[0]["Carbohydrates"]
    for i in range(len(carb_data)):
        date = date_column[i]
        if date not in carb_dict.keys():
            carb_dict[date] = carb_data[i]
        else:
            carb_dict[date] += carb_data[i]
    carb_data = [carb_dict[key] for key in sorted(carb_dict.keys())]

    fat_dict = dict()
    fat_data = data[0]["Fat"]
    for i in range(len(fat_data)):
        date = date_column[i]
        if date not in fat_dict.keys():
            fat_dict[date] = fat_data[i]
        else:
            fat_dict[date] += fat_data[i]
    fat_data = [fat_dict[key] for key in sorted(fat_dict.keys())]

    meal_dict = dict()
    meal_dict["date"] = sorted(list(set(data[0]["date"])))
    for meal in meal_set:
        meal_dict[meal] = data[0][meal]

    food_items = []
    ingredient_items = []
    food_ids = dict()
    ingredient_ids = dict()

    num_days = 1
    meal_list = []
    new_cnt = 1
    new_day = False
    tmp = []
    for i in range(len(food_data)):
        meal = meal_column[i]
        if meal in meal_list:
            num_days += 1
            meal_list = [meal]
            new_day = True
            food_items.append(tmp)
            tmp = []
        else:
            meal_list.append(meal)

        food_ = food_data[i].strip()
        tmp.append(food_)

        if food_ not in food_ids:
            if new_day:
                new_cnt += 1
                new_day = False
            food_ids[food_] = len(food_ids.keys())

    for i in range(len(ingredient_data)):
        meal = meal_column[i]
        dish = ingredient_data[i]
        i_list = dish[1:-1].split('},')
        tmp = []
        for item in i_list:
            if ':' in item:
                item = item.split(":")[1].strip(' \'}]')

            ingredient_ids[item] = len(food_ids.keys())
            tmp.append(item)

        ingredient_items.append(tmp)

    # New recipe frequency
    new_freq = round(float(new_cnt)/num_days,2)

    t = URIRef('http://purl.obolibrary.org/obo/PATO_0002360/t')
    fr1 = URIRef('http://semanticscience.org/resource/SIO_001367/fr1')
    snr = URIRef('http://idea.rpi.edu/heals/phkg/SelectingNewRecipes/snr')
    phkg.add((t,RDF.type,tendency))
    phkg.add((t,has_attribute, snr))
    phkg.add((user,has_attribute,t))
    phkg.add((fr1,RDF.type,frequency))
    phkg.add((t,has_attribute,fr1))
    phkg.add((fr1,has_value,Literal(new_freq,datatype=XSD.float)))

    food_set = set(food_ids.keys())
    food_list = list(food_set)
    food_data = food_items

    ingredient_set = set(ingredient_ids.keys())
    ingredient_list = list(ingredient_set)
    ingredient_data = ingredient_items

    food = food_list[0]

    series_dict_f = build_time_series_daily(food_set,food_data)

    series_dict_im = build_time_series_meal(ingredient_set,ingredient_data)
    series_dict_id = build_time_series_daily(ingredient_set,ingredient_data)

    # Find recipe preferences
    max_freq = 0
    fav_food = None
    for i in range(len(food_list)):
        food, ratio = find_preference(food_list[i],series_dict_f[food_list[i]])

        if ratio > max_freq:
            fav_food = food
            max_freq = ratio

    for i in range(len(ingredient_list)): # Ask about this (day vs meal granularity)
        food, ratio = find_preference(ingredient_list[i],series_dict_id[ingredient_list[i]])

        if ratio > max_freq:
            fav_food = food
            max_freq = ratio

    fp1 = URIRef('http://purl.obolibrary.org/obo/NBO_0000141/fp1')
    fr2 = URIRef('http://semanticscience.org/resource/SIO_001367/fr2')
    f1 = URIRef('http://purl.obolibrary.org/obo/NCIT_C62695/f1')

    phkg.add((fp1, RDF.type, FoodPreference))
    phkg.add((user, has_attribute, fp1))
    phkg.add((fr2, RDF.type, frequency))
    phkg.add((fp1, has_attribute, fr2))
    phkg.add((fr2, has_value, Literal(max_freq,datatype=XSD.float)))
    phkg.add((f1, RDF.type, Food))
    phkg.add((fp1, has_attribute, f1))
    phkg.add((f1, RDFS.label, Literal(fav_food,datatype=XSD.string)))

    # Find relationships between food items
    ingredient1 = ingredient_list[0]
    ingredient2 = ingredient_list[1]
    freq, freq_ = find_relationship(attr,ingredient1,ingredient2,series_dict_im,letter_map_list,alpha,alpha_sizes,age,activity_level,TW)

    if freq != 0:

        frel1 = URIRef('http://purl.obolibrary.org/obo/NCIT_C25648/frel1')
        f2 = URIRef('http://purl.obolibrary.org/obo/NCIT_C62695/f2')
        f3 = URIRef('http://purl.obolibrary.org/obo/NCIT_C62695/f3')
        fr3 = URIRef('http://semanticscience.org/resource/SIO_001367/fr3')

        phkg.add((frel1, RDF.type, Relationship))
        phkg.add((user, has_attribute, frel1))
        phkg.add((f2, RDF.type, Food))
        phkg.add((f3, RDF.type, Food))
        phkg.add((frel1, has_participant, f2))
        phkg.add((frel1, has_participant, f3))
        phkg.add((f2, RDFS.label, Literal(ingredient1,datatype=XSD.string)))
        phkg.add((f3, RDFS.label, Literal(ingredient2,datatype=XSD.string)))
        phkg.add((f3, is_causally_related-to, f2))
        phkg.add((fr3, RDF.type, frequency))
        phkg.add((frel1, has_attribute, fr3))
        phkg.add((fr3, has_value, Literal(freq,datatype=XSD.float)))

    # Find food item sequences in data
    food = food_list[0]
    find_food_sequences(food,series_dict_f[food])

    # Consistent carbohydrate evaluation
    attr = "MyFitnessPalMeals"
    letter_map = dict()
    for i in range(alpha_size):
        letter_map[alphabet[i]] = i+1
    guideline = "DG02"
    TW = "weeks"
    relative_fix(attr,meal_dict,"weeks","DG02",letter_map)

    # High carb, low fat
    attr = "MyFitnessPal"
    tw = len(carb_data)
    guideline = "DG05"
    singular_TW = "week"
    general_ADA_summary(meal_dict,attr,"highcarblowfat",tw,guideline,TW,singular_TW,letter_map)

    # Goal Evaluation w/ Qualifier
    goals = []
    key_list = ["Carbohydrate Intake","Fat Intake"]
    nutrient_data = [carb_data,fat_data]
    summarizer_7 = None
    start_day = 0
    end_day = len(carb_data)-1
    goals = [calorie_data]*2

    summarizers, average = generateSESTWQ(attr,key_list,nutrient_data,summarizer_7,start_day,end_day,alpha,alpha_sizes,letter_map_list,alphabet_list,TW,age,activity_level,goals=goals,constraint=True)

    highcarb = "true" if summarizers[0] == "reached" else "false"
    lowfat = "true" if summarizers[1] == "reached" else "false"

    ng = URIRef('http://idea.rpi.edu/heals/phkg/NutrientIntakeGoal/ng')
    ir = URIRef('http://purl.obolibrary.org/obo/NCIT_C25648/ir')
    hc = URIRef('http://idea.rpi.edu/heals/phkg/HighCarb/hc')
    lf = URIRef('http://idea.rpi.edu/heals/phkg/LowFat/lf')
    fr6 = URIRef('http://semanticscience.org/resource/SIO_001367/fr6')

    phkg.add((ng, RDF.type, NutrientIntakeGoal))
    phkg.add((user, has_attribute, ng))
    phkg.add((ir, RDF.type, Relationship))
    phkg.add((ng, is_related_to, ir))
    phkg.add((hc, RDF.type, HighCarb))
    phkg.add((lf, RDF.type, LowFat))
    phkg.add((ir, has_participant, hc))
    phkg.add((ir, has_participant, lf))
    phkg.add((hc, has_value, Literal(highcarb,datatype=XSD.boolean)))
    phkg.add((lf, has_value, Literal(lowfat,datatype=XSD.boolean)))
    phkg.add((hc, is_related_to, lf))
    phkg.add((fr6, RDF.type, frequency))
    phkg.add((ir, has_attribute, fr6))
    phkg.add((fr6, has_value, Literal(average,datatype=XSD.float)))

    # Weekday-Based Pattern + Goal (right now, only for Sunday)
    date_column = list(sorted(set(date_column)))
    for i in range(len(date_column)):
        date = date_column[i]
        date = date.split('/')
        date = datetime.datetime(int(date[2]),int(date[0]),int(date[1]))
        date_column[i] = date

    date_column = [weekday_dict[date.weekday()+1] for date in date_column]
    date_column = [date for date in date_column if date == "Sunday"]
    summarizers, average_list = generateDB(attr,key_list,nutrient_data,summarizer_7,start_day,end_day,alpha,alpha_sizes,letter_map_list,alphabet_list,tw,TW,age,activity_level,date_column,goals=goals,constraint=True)

    average = max(average_list)
    summarizers = summarizers[0][average_list.index(average)]

    highcarb = "true" if summarizers[0] == "reach" else "false"
    lowfat = "true" if summarizers[1] == "reach" else "false"


    w = URIRef('http://purl.obolibrary.org/obo/NCIT_C86936/w')
    fr5 = URIRef('http://semanticscience.org/resource/SIO_001367/fr5')

    phkg.add((w, RDF.type, Sunday))
    phkg.add((ng, has_time_boundary, w))
    phkg.add((fr5, RDF.type, frequency))
    phkg.add((ng, has_frequency, fr5))
    phkg.add((fr5, has_value, Literal(average,datatype=XSD.float)))

    # Cluster-Based Pattern + Goal
    tw = 7
    letter_map_list = [{'a' : 0,
                        'b' : 1}]*len(key_list)
    alpha_sizes = [2]*2
    sax_list, tw_sax_list, binary_list = build_sax_list(key_list,nutrient_data,goals,alphabet,alpha_size)
    summarizers, average = generateCB(attr,[attr],key_list,sax_list[0],tw_sax_list,sax_list,nutrient_data,letter_map_list,alpha_sizes,alpha,tw,TW,age,activity_level,constraint=True)
    if summarizers != None:

        tp = URIRef('http://purl.obolibrary.org/obo/NCIT_C54156/tp')
        nw = URIRef('http://semanticscience.org/resource/SIO_001354/nw')
        cf = URIRef('http://semanticscience.org/resource/SIO_001367/cf')
        ng1 = URIRef('http://idea.rpi.edu/heals/phkg/NutrientIntakeGoal/ng1')
        r = URIRef('http://purl.obolibrary.org/obo/NCIT_C25648/r')
        hc1 = URIRef('http://idea.rpi.edu/heals/phkg/HighCarb/hc1')
        lf1 = URIRef('http://idea.rpi.edu/heals/phkg/LowFat/lf1')
        fr4 = URIRef('http://semanticscience.org/resource/SIO_001367/fr4')

        phkg.add((tp, RDF.type, Prediction))
        phkg.add((user, has_attribute, tp))
        phkg.add((nw, RDF.type, Week))
        phkg.add((tp, is_related_to, nw))
        phkg.add((ng1, RDF.type, NutrientIntakeGoal))
        phkg.add((ng1, is_related_to, r))
        phkg.add((r, RDF.type, Relationship))
        phkg.add((hc1, RDF.type, HighCarb))
        phkg.add((lf1, RDF.type, LowFat))
        phkg.add((r, has_participant, hc1))
        phkg.add((r, has_participant, lf1))
        phkg.add((hc1, has_value, Literal(highcarb,datatype=XSD.boolean)))
        phkg.add((lf1, has_value, Literal(lowfat,datatype=XSD.boolean)))
        phkg.add((tp, is_related_to, ng1))
        phkg.add((ng1, has_frequency, cf))
        phkg.add((cf, RDF.type, frequency))
        phkg.add((cf, has_value, Literal(average,datatype=XSD.float)))

    # Find guideline behavioral sequences in data
    # TODO: Fix the protoform in analyze_patterns
    find_behavior_sequences(binary_list)

    # Anomaly detection
    from saxpy.hotsax import find_discords_hotsax
    discords = find_discords_hotsax(nutrient_data[1])
    if len(discords) > 0:
        input("Fix this once anomaly is found.")

        ia = URIRef('http://idea.rpi.edu/heals/phkg/IntakeAnomaly/ia')
        ci = URIRef('http://www.enpadasi.eu/ontology/release/v1/ons/ONS_0000023/ci')
        th = URIRef('http://semanticscience.org/resource/SIO_000130/th')
        bm = URIRef('http://purl.obolibrary.org/obo/NCIT_C80249/bm')
        t = URIRef('http://purl.obolibrary.org/obo/NCIT_C41134/t')

        phkg.add((ia, RDF.type, IntakeAnomaly))
        phkg.add((user, has_attribute, ia))
        phkg.add((ci, RDF.type, CarbohydrateIntake))
        phkg.add((ia, refers_to, ci))
        phkg.add((th, RDF.type, pattern))
        phkg.add((ci, is_described_by, th))
        phkg.add((bm, RDF.type, Breakfast))
        phkg.add((t, RDF.type, Today))
        phkg.add((ia, has_temporal_boundary, bm))
        phkg.add((ia, has_temporal_boundary, t))

    # Long-term trend
    long_term_trend(binary_list)

    rdf_file.write(phkg.serialize(format="trig").decode('utf-8'))
    rdf_file.close()

if __name__ == "__main__":
    main()
