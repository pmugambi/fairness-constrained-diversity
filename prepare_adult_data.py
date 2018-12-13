def read_data():
    with open("./data/adult.data.txt") as a_data:
        lines = a_data.readlines()
        return lines


def assign_work_class_num(work_class):
    work_classes = ["private", "self-emp-not-inc", "self-emp-inc", "federal-gov", "local-gov", "state-gov",
                    "without-pay", "never-worked"]
    if work_class in work_classes:
        return work_classes.index(work_class)
    else:
        return -1


def assign_marital_status_num(marital_status):
    marital_statuses = ["married-civ-spouse", "divorced", "never-married", "separated", "widowed",
                        "married-spouse-absent", "married-af-spouse"]
    if marital_status in marital_statuses:
        return marital_statuses.index(marital_status)
    else:
        return -1


def assign_occupations_num(occupation):
    occupations = ["tech-support", "craft-repair", "other-service", "sales", "exec-managerial", "prof-specialty",
                   "handlers-cleaners", "machine-op-inspct", "adm-clerical", "farming-fishing", "transport-moving",
                   "priv-house-serv", "protective-serv", "armed-forces"]
    if occupation in occupations:
        return occupations.index(occupation)
    else:
        return -1


def assign_relationship_num(relationship):
    relationships = ["wife", "own-child", "husband", "not-in-family", "other-relative", "unmarried"]
    if relationship in relationships:
        return relationships.index(relationship)
    else:
        return -1


def assign_race_num(race):
    races = ["white", "asian-pac-islander", "amer-indian-eskimo", "other", "black"]
    if race in races:
        return races.index(race)
    else:
        return -1


def assign_gender_num(gender):
    genders = ["female", "male"]
    if gender in genders:
        return genders.index(gender)
    else:
        return -1


def assign_native_country_num(native_country):
    countries = ["united-states", "cambodia", "england", "puerto-rico", "canada", "germany",
                 "outlying-us(guam-usvi-etc)", "india", "japan", "greece", "south", "china", "cuba", "iran", "honduras",
                 "philippines", "italy", "poland", "jamaica", "vietnam", "mexico", "portugal", "ireland", "france",
                 "dominican-republic", "laos", "ecuador", "taiwan", "haiti", "columbia", "hungary", "guatemala",
                 "nicaragua", "scotland", "thailand", "yugoslavia", "el-salvador",
                 "trinadad&tobago", "peru", "hong", "holand-netherlands"]

    if native_country in countries:
        return countries.index(native_country)
    else:
        return -1


def assign_salary_num(salary):
    salaries = [">50k", "<=50k"]

    if salary in salaries:
        return salaries.index(salary)
    else:
        return -1


def process():
    processed_data = []
    data = read_data()
    for i in xrange(0, 3):
        line = data[i].split(" ")
        print line
        age = float(line[0].replace(",", ""))
        work_class = line[1].replace(",", "").lower()
        work_class_num = assign_work_class_num(work_class)
        fnlwgt = float(line[2].replace(",", ""))
        education = line[3].replace(",", "").lower()
        education_num = float(line[4].replace(",", ""))
        marital_status = line[5].replace(",", "").lower()
        marital_status_num = assign_marital_status_num(marital_status)
        occupation = line[6].replace(",", "").lower()
        occupation_num = assign_occupations_num(occupation)
        relationship = line[7].replace(",", "").lower()
        relationship_num = assign_relationship_num(relationship)
        race = line[8].replace(",", "").lower()
        race_num = assign_race_num(race)
        gender = line[9].replace(",", "").lower()
        gender_num = assign_gender_num(gender)

        capital_gain = float(line[10].replace(",", ""))
        capital_loss = float(line[11].replace(",", ""))
        hours_per_week = float(line[12].replace(",", ""))
        native_country = line[13].replace(",", "").lower()
        # print native_country
        native_country_num = assign_native_country_num(native_country)
        # print native_country_num

        salary = line[14].replace("\n", "").lower()
        salary_num = assign_salary_num(salary)

        num_row = [age, work_class_num, fnlwgt, education_num, marital_status_num,
                   occupation_num, relationship_num, race_num, gender_num, capital_gain, capital_loss, hours_per_week,
                   native_country_num, salary_num]

        print num_row
        processed_data.append(num_row)
    return processed_data

process()


