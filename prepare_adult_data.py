from collections import Counter


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


def process(count, columns=None):
    if columns is None:
        columns = ["all"]

    print "columns = ", columns
    processed_data = []
    data = read_data()
    # print "data = ", data
    # for i in xrange(0, count):
    for i in xrange(0, len(data)):
        line = data[i].split(" ")
        # print line
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
        # print "occupation and occupation_num = ", occupation, occupation_num
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
        native_country_num = assign_native_country_num(native_country)

        salary = line[14].replace("\n", "").lower()
        salary_num = assign_salary_num(salary)

        all = "all"
        if all in columns:
            # num_row = [age, work_class_num, fnlwgt, education_num, marital_status_num,
            #            occupation_num, relationship_num, race_num, gender_num, capital_gain, capital_loss,
            #            hours_per_week,
            #            native_country_num, salary_num]

            num_row = [age, work_class, fnlwgt, education, marital_status, occupation, relationship, race, gender,
                       capital_gain, capital_loss, hours_per_week, native_country, salary]
        else:
            num_row = []
            for column in columns:
                if column.lower() == "age":
                    num_row.append(age)
                if column.lower() == "work_class":
                    num_row.append(work_class)
                if column.lower() == "fnlwgt":
                    num_row.append(fnlwgt)
                if column.lower() == "education":
                    num_row.append(education)
                if column.lower() == "education_num":
                    num_row.append(education_num)
                if column.lower() == "marital_status":
                    num_row.append(marital_status)
                if column.lower() == "occupation":
                    num_row.append(occupation)
                if column.lower() == "relationship":
                    num_row.append(relationship)
                if column.lower() == "race":
                    num_row.append(race)
                if column.lower() == "gender":
                    num_row.append(gender)
                if column.lower() == "capital_gain":
                    num_row.append(capital_gain)
                if column.lower() == "capital_loss":
                    num_row.append(capital_loss)
                if column.lower() == "hours_per_week":
                    num_row.append(hours_per_week)
                if column.lower() == "native_country":
                    num_row.append(native_country)
                if column.lower() == "salary":
                    num_row.append(salary)
                if column.lower() == "gender_num":
                    num_row.append(gender_num)
                if column.lower() == "work_class_num":
                    num_row.append(work_class_num)
                if column.lower() == "marital_status_num":
                    num_row.append(marital_status_num)
                if column.lower() == "occupation_num":
                    num_row.append(occupation_num)
                if column.lower() == "relationship_num":
                    num_row.append(relationship_num)
                if column.lower() == "race_num":
                    num_row.append(race_num)
                if column.lower() == "native_country_num":
                    num_row.append(native_country_num)
                # else:
                #     print "column ", column, " not found"

        # print num_row
        processed_data.append(num_row)
    # print "processed data = ", processed_data
    return processed_data


def obtain_sensitive_attributes_columns(sensitive_attributes):
    cols = []
    for sa in sensitive_attributes:
        if sa.lower() == "gender":
            cols.append(8)
        if sa.lower() == "race":
            cols.append(7)
        if sa.lower() == "marital_status":
            cols.append(4)
    return cols


def compute_proportions(data, sa_index):
    rows = []
    for i in data:
        rows.append(i[sa_index])
    return rows

# process(columns=["age", "education", "occupation"])
# process()

