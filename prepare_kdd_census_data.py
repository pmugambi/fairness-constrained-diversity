import prepare_adult_data as pad


def read_data():
    with open("./data/kdd_census/census-income.data") as a_data:
        lines = a_data.readlines()
        return lines


def read_test_data():
    with open("./data/kdd_census/census-income.test") as a_data:
        lines = a_data.readlines()
        print len(lines)
        return lines


def obtain_sensitive_attribute_column(sensitive_attribute):
    cols = []
    if sensitive_attribute.lower() == "gender":
        cols.append(12)
    if sensitive_attribute.lower() == "race":
        cols.append(10)
    if sensitive_attribute.lower() == "marital_status":
        cols.append(7)
    return cols


def clean_rows(l):
    data = read_data()
    processed_data = []

    # for i in xrange(0, len(data)):
    for i in xrange(0, l):
        line = data[i].split(",")

        age = float(line[0])
        work_class = line[1].strip().lower()
        industry_code = float(line[2].strip())
        occupation_code = float(line[3].strip())
        education = line[4].strip().lower()
        hourly_wage = float(line[5].strip())
        enroll_in_edu_inst = line[6].strip().lower()
        marital_status = line[7].strip().lower()
        industry = line[8].strip().lower()
        occupation = line[9].strip().lower()
        race = line[10].strip().lower()
        hispanic_origin = line[11].strip().lower()
        gender = line[12].strip().lower()
        labour_union_member = line[13].strip().lower()
        unemployment_reason = line[14].strip().lower()
        full_parttime_employment = line[15].strip().lower()
        capital_gain = float(line[16].strip())
        capital_loss = float(line[17].strip())
        dividends = float(line[18].strip())
        tax_filer = line[19].strip().lower()
        previous_residence_region = line[20].strip().lower()
        previous_residence_state = line[21].strip().lower()
        detailed_household_and_family_info = line[22].strip().lower()
        detailed_household_summary = line[23].strip().lower()
        # add weight variable
        weight_value = float(line[24].strip())
        migration_code_change_msa = line[25].strip().lower()
        migration_code_change_reg = line[26].strip().lower()
        migration_code_move_reg = line[27].strip().lower()
        in_house_past_year = line[28].strip().lower()
        migration_prev_res = line[29].strip().lower()
        persons_working_for_employer = line[30].strip()
        family_members_under_18 = line[31].strip().lower()
        father_cob = line[32].strip().lower()
        mother_cob = line[33].strip().lower()
        self_cob = line[34].strip().lower()
        citizenship = line[35].strip().lower()
        self_employed_bo = float(line[36].strip())
        veteran_questionairre_fill = line[37].strip().lower()
        veteran_benefits = float(line[38].strip())
        weeks_worked_in_year = float(line[39].strip())
        year = float(line[40].strip())

        clean_row = [age, work_class, industry_code, occupation_code, education, hourly_wage, enroll_in_edu_inst,
                     marital_status, industry, occupation, race, hispanic_origin, gender, labour_union_member,
                     unemployment_reason, full_parttime_employment, capital_gain, capital_loss, dividends, tax_filer,
                     previous_residence_region, previous_residence_state, detailed_household_and_family_info,
                     detailed_household_summary, migration_code_change_msa, migration_code_change_reg,
                     migration_code_move_reg, in_house_past_year, migration_prev_res, persons_working_for_employer,
                     family_members_under_18, father_cob, mother_cob, self_cob, citizenship, self_employed_bo,
                     veteran_questionairre_fill, veteran_benefits, weeks_worked_in_year, year]
        processed_data.append(clean_row)
    return processed_data


def process(count, column_name):
    data = read_data()
    processed_data = []

    for i in xrange(0, count):
    # for i in xrange(0, len(data)):
        line = data[i].split(",")
        # print line, len(line)
        age = float(line[0])
        work_class = line[1].strip().lower()
        industry_code = float(line[2].strip())
        occupation_code = float(line[3].strip())
        marital_status = line[7].strip().lower()
        gender = line[12].strip().lower()
        race = line[10].strip().lower()
        capital_gain = float(line[16].strip())
        capital_loss = float(line[17].strip())
        dividends = float(line[17].strip())
        weeks_worked_in_year = float(line[39].strip())

        # print type(age), age, type(work_class), work_class, industry_code, occupation_code
        num_row = []
        if column_name.lower() == "age":
            num_row.append(age)
        if column_name.lower() == "work_class":
            num_row.append(work_class)
        if column_name.lower() == "industry_code":
            num_row.append(industry_code)
        if column_name.lower() == "occupation_code":
            num_row.append(occupation_code)
        if column_name.lower() == "capital_gain":
            num_row.append(capital_gain)
        if column_name.lower() == "capital_loss":
            num_row.append(capital_loss)
        if column_name.lower() == "dividends":
            num_row.append(dividends)
        if column_name.lower() == "gender_num":
            num_row.append(pad.assign_gender_num(gender))
        if column_name.lower() == "race_num":
            num_row.append(pad.assign_race_num(race))
        if column_name.lower() == "marital_status_num":
            num_row.append(pad.assign_marital_status_num(marital_status))
        if column_name.lower() == "gender":
            num_row.append(gender)
        if column_name.lower() == "year_weeks":
            num_row.append(weeks_worked_in_year)

        processed_data.append(num_row)
    # print "processed data  = ", processed_data
    return processed_data


def draw_histogram(data):
    import numpy as np
    import matplotlib.mlab as mlab
    import matplotlib.pyplot as plt
    from scipy.stats import norm


    # mu, sigma = 100, 15
    # x = mu + sigma * np.random.randn(10000)
    d = np.asmatrix(np.array(data))
    print "data as a matrix = ", d

    # the histogram of the data
    n, bins, patches = plt.hist(d, 10, normed=1, facecolor='green', alpha=0.75)
    # n, bins, patches = plt.hist(d, 10)

    print "bins = ", bins

    mu = np.mean(d, axis=0)[0][0]
    sigma = np.std(d, axis=0)[0][0]

    (mu, sigma) = norm.fit(data)

    print "mean, std = ", mu, sigma

    # add a 'best fit' line
    y = mlab.normpdf(bins, mu, sigma)
    l = plt.plot(bins, y, 'r--', linewidth=1)

    plt.xlabel('Smarts')
    plt.ylabel('Probability')
    plt.title(r'$\mathrm{Histogram\ of\ Capital Gain:}\ \mu='+str(mu)+',\ \sigma='+str(sigma)+'$')
    # plt.axis([-5, 55, 0, 0.1])
    plt.grid(False)

    plt.show()

# data = process(40000, "capital_gain")
# draw_histogram(data)

# d = read_test_data()
# print d[0]

# df = pd.read_csv("./data/kdd_census/census-income.test", sep=',')
# df = pd.read_csv("./data/kdd_census/census-income.data", sep=',')
# print "length of df = ", len(df)
# print "correlation = ", df.corr(method='pearson')


