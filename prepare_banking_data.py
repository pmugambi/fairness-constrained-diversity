def read_data():
    with open("./data/bank_marketing/bank.csv") as a_data:
        lines = a_data.readlines()
        return lines


def marital_status_number(marital_status):
    marital_statuses = ["married", "divorced", "single"]
    return marital_statuses.index(marital_status)


def job_number(job):
    jobs = ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'services',
            'student', 'technician', 'unemployed', 'unknown']
    return jobs.index(job)


def clean_rows(l):
    data = read_data()

    processed_rows = []
    for i in xrange(1, l):
        line = data[i]
        line_readings = line.split(';')
        # print line_readings

        age = int(line_readings[0])
        job = line_readings[1].replace('"', '')
        marital = line_readings[2].replace('"', '')
        education = line_readings[3].replace('"', '')
        default = line_readings[4].replace('"', '')
        balance = int(line_readings[5])
        housing = line_readings[6].replace('"', '')
        loan = line_readings[7].replace('"', '')
        contact = line_readings[8].replace('"', '')
        day = int(line_readings[9])
        month = line_readings[10].replace('"', '')
        duration = int(line_readings[11])
        campaign = int(line_readings[12])
        pdays = int(line_readings[13])
        previous = int(line_readings[14])
        poutcome = line_readings[15].replace('"', '')
        y = line_readings[16].replace('"', '').replace('\n', '')
        # print balance

        # print age, job, marital, education, default, balance, housing, loan, contact, day, month, duration, campaign, \
        #     pdays, previous, poutcome, y

        num_row = [age, job, marital, education, default, balance, housing, loan, contact, day, month, duration,
                   campaign, pdays, previous, poutcome, y]
        processed_rows.append(num_row)
    return processed_rows


def process(l, column_name):
    data = read_data()
    processed_rows = []
    for i in xrange(1, l):
        line = data[i]
        line_readings = line.split(';')
        # print line_readings

        age = int(line_readings[0])
        job = line_readings[1].replace('"', '')
        marital = line_readings[2].replace('"', '')
        education = line_readings[3].replace('"', '')
        default = line_readings[4].replace('"', '')
        balance = int(line_readings[5])
        housing = line_readings[6].replace('"', '')
        loan = line_readings[7].replace('"', '')
        contact = line_readings[8].replace('"', '')
        day = int(line_readings[9])
        month = line_readings[10].replace('"', '')
        duration = int(line_readings[11])
        campaign = int(line_readings[12])
        pdays = int(line_readings[13])
        previous = int(line_readings[14])
        poutcome = line_readings[15].replace('"', '')
        y = line_readings[16].replace('"', '').replace('\n', '')

        # print age, job, marital, education, default, balance, housing, loan, contact, day, month, duration, campaign, \
        #     pdays, previous, poutcome, y

        num_row = []

        if column_name == "balance":
            num_row.append(balance)
        if column_name == "age":
            num_row.append(age)
        if column_name == "marital":
            num_row.append(marital)
        if column_name == "marital_num":
            num_row.append(marital_status_number(marital))
        if column_name == "job_num":
            num_row.append(job_number(job))
        processed_rows.append(num_row)
    return processed_rows


# print "total lines = ", len(read_data())
# process(read_data(),"balance")
