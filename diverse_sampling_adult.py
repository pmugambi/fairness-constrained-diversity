import numpy as np
import prepare_adult_data as pad
import prepare_kdd_census_data as pkcd
import prepare_banking_data as pbd
import gmm as gmm
import gms as gms
import math
import helpers as h
from scipy.spatial.distance import pdist, squareform, euclidean
import matplotlib.pyplot as plt


def compute_highest_difference(data):
    min_v = min(data)[0]
    max_v = max(data)[0]

    print "min value = ", min_v
    print "max value = ", max_v

    print "max_diff = ", max_v - min_v

    return max_v - min_v


def get_highest_value(data):
    return max(data)[0]


def sample_mixed_data_diverse_k(con_data, cat_data, norm_val, k, weight_1, weight_2, algorithm):
    con_d = np.asmatrix(np.array(con_data))
    cat_d = np.asmatrix(np.array(cat_data))
    if algorithm == "max_min":
        top_k = gmm.greedy_diverse_mod(con_d, cat_d, norm_val, k, weight_1, weight_2)
    elif algorithm == "max_sum":
        top_k = gms.greedy_diverse_mod(con_d, cat_d, norm_val, k, weight_1, weight_2)
    # print top_k
    else:
        top_k = None
    return top_k


def evaluate_fairness(data, sample, fairness_attribute_index):
    sample_sensitive_attributes_values = []
    for index in sample:
        data_line = data[index]
        sample_fairness_attribute_value = data_line[fairness_attribute_index]
        sample_sensitive_attributes_values.append(sample_fairness_attribute_value)

    # print "sample sensitive attribute values = ", sample_sensitive_attributes_values
    return sample_sensitive_attributes_values


def normalize_by_value(normalization_method, data):
    if normalization_method == "max_diff":
        norm_by = compute_highest_difference(data)
    elif normalization_method == "max_val":
        norm_by = get_highest_value(data)
    else:
        norm_by = 1  # maintaining values as they were. No normalization
    return norm_by


def sample_on(perc, data_set, x, con_diversification_attribute, cat_diversification_attribute, weight_1, weight_2,
              normalization_method, algorithm):
    total_data = []
    con_data = None
    cat_data = None

    if data_set.lower() == "adult":
        total_data = pad.process(x)
        con_data = pad.process(x, con_diversification_attribute)  # obtain data based on diversification attribute
        cat_data = pad.process(x, cat_diversification_attribute)
        # sample = sample_mixed_data_diverse_k(data, None, norm_by, perc, weight_1, weight_2, algorithm)
    elif data_set.lower() == "census":
        total_data = pkcd.clean_rows(x)
        con_data = pkcd.process(x, con_diversification_attribute)  # obtain data based on diversification attribute
        cat_data = pkcd.process(x, cat_diversification_attribute)
    elif dataset.lower() == "bank":
        total_data = pbd.clean_rows(x)
        con_data = pbd.process(x, con_diversification_attribute)
        cat_data = pbd.process(x, cat_diversification_attribute)

    norm_by = normalize_by_value(normalization_method, con_data)
    sample = sample_mixed_data_diverse_k(con_data, cat_data, norm_by, perc, weight_1, weight_2, algorithm)

    return total_data, sample, con_diversification_attribute, cat_diversification_attribute


def compute_gender_proportions(data_set, total_data, sample):
    if data_set.lower() == "adult":
        fairness_attribute = pad.obtain_sensitive_attribute_column("gender")[0]
    else:
        fairness_attribute = pkcd.obtain_sensitive_attribute_column("gender")[0]

    sample_sensitive_attribute_values = evaluate_fairness(total_data, sample, fairness_attribute)

    f_values = ["female"]
    m_values = ["male"]

    total_female_prop = float(
        h.my_counter(pad.compute_proportions(total_data, fairness_attribute), f_values)) / len(total_data)
    total_male_prop = float(h.my_counter(pad.compute_proportions(total_data, fairness_attribute), m_values)) / len(
        total_data)

    sample_female_prop = float(h.my_counter(sample_sensitive_attribute_values, f_values)) / len(sample)
    sample_male_prop = float(h.my_counter(sample_sensitive_attribute_values, m_values)) / len(sample)

    props_x = [total_male_prop, total_female_prop, sample_male_prop, sample_female_prop]

    print "gender_2 proportions = ", props_x
    return props_x


def compute_racial_proportions(data_set, total_data, sample):
    if data_set.lower() == "adult":
        fairness_attribute = pad.obtain_sensitive_attribute_column("race")[0]
    else:
        fairness_attribute = pkcd.obtain_sensitive_attribute_column("race")[0]
    sample_sensitive_attribute_values = evaluate_fairness(total_data, sample, fairness_attribute)

    w_values = ["white"]
    b_values = ["black"]
    n_values = ["amer-indian-eskimo", "amer indian aleut or eskimo"]
    a_values = ["asian-pac-islander", "asian or pacific islander"]
    o_values = ["others", "other"]

    total_white_prop = float(
        h.my_counter(pad.compute_proportions(total_data, fairness_attribute), w_values)) / len(total_data)
    total_black_prop = float(h.my_counter(pad.compute_proportions(total_data, fairness_attribute), b_values)) / len(
        total_data)
    total_asian_prop = float(h.my_counter(pad.compute_proportions(total_data, fairness_attribute), a_values)) / len(
        total_data)
    total_other_prop = float(h.my_counter(pad.compute_proportions(total_data, fairness_attribute), o_values)) / len(
        total_data)
    total_native_a_prop = float(h.my_counter(pad.compute_proportions(total_data, fairness_attribute), n_values)) / len(
        total_data)

    sample_white_prop = float(h.my_counter(sample_sensitive_attribute_values, w_values)) / len(sample)
    sample_black_prop = float(h.my_counter(sample_sensitive_attribute_values, b_values)) / len(sample)
    sample_asian_prop = float(h.my_counter(sample_sensitive_attribute_values, a_values)) / len(sample)
    sample_other_prop = float(h.my_counter(sample_sensitive_attribute_values, o_values)) / len(sample)
    sample_native_a_prop = float(h.my_counter(sample_sensitive_attribute_values, n_values)) / len(sample)

    props_x = [total_white_prop, total_black_prop, total_asian_prop, total_other_prop, total_native_a_prop,
               sample_white_prop, sample_black_prop, sample_asian_prop, sample_other_prop, sample_native_a_prop]

    print "race proportions = ", props_x
    return props_x


def compute_marital_proportions(data_set, total_data, sample):
    if data_set.lower() == "adult":
        fairness_attribute = pad.obtain_sensitive_attribute_column("marital_status")[0]
    elif dataset.lower() == "census":
        fairness_attribute = pkcd.obtain_sensitive_attribute_column("marital_status")[0]
    else:  # remember to remove this hard-coding for bank dataset
        fairness_attribute = 2

    nevermarried_values = ["never-married", "never married", "single"]  # added 'single' due to the banking dataset
    marriedciv_values = ["married-civilian spouse present", "married-civ-spouse",
                         "married"]  # added 'married' due to the banking dataset
    marriedabsentspouse_values = ["married-spouse absent", "married-spouse-absent"]
    separated_values = ["separated"]
    divorced_values = ["divorced"]
    widowed_values = ["widowed"]
    marriedaf_values = ["married-a f spouse present", "married-af-spouse"]

    sample_sensitive_attribute_values = evaluate_fairness(total_data, sample, fairness_attribute)

    total_married_civ_spouse_prop = float(
        h.my_counter(pad.compute_proportions(total_data, fairness_attribute), marriedciv_values)) / len(total_data)
    total_divorced_prop = float(
        h.my_counter(pad.compute_proportions(total_data, fairness_attribute), divorced_values)) / len(
        total_data)
    total_never_married_prop = float(
        h.my_counter(pad.compute_proportions(total_data, fairness_attribute), nevermarried_values)) / len(
        total_data)
    total_separated_prop = float(
        h.my_counter(pad.compute_proportions(total_data, fairness_attribute), separated_values)) / len(
        total_data)
    total_widowed_prop = float(
        h.my_counter(pad.compute_proportions(total_data, fairness_attribute), widowed_values)) / len(
        total_data)
    total_married_spouse_absent_prop = float(
        h.my_counter(pad.compute_proportions(total_data, fairness_attribute), marriedabsentspouse_values)) / len(
        total_data)
    total_married_af_spouse_prop = float(
        h.my_counter(pad.compute_proportions(total_data, fairness_attribute), marriedaf_values)) / len(
        total_data)

    sample_married_civ_spouse_prop = float(h.my_counter(sample_sensitive_attribute_values, marriedciv_values)) / len(
        sample)
    sample_divorced_prop = float(h.my_counter(sample_sensitive_attribute_values, divorced_values)) / len(sample)
    sample_never_married_prop = float(h.my_counter(sample_sensitive_attribute_values, nevermarried_values)) / len(
        sample)
    sample_separated_prop = float(h.my_counter(sample_sensitive_attribute_values, separated_values)) / len(sample)
    sample_widowed_prop = float(h.my_counter(sample_sensitive_attribute_values, widowed_values)) / len(sample)
    sample_married_spouse_absent_prop = float(
        h.my_counter(sample_sensitive_attribute_values, marriedabsentspouse_values)) / len(sample)
    sample_married_af_spouse_prop = float(h.my_counter(sample_sensitive_attribute_values, marriedaf_values)) / len(
        sample)

    props_x = [total_married_civ_spouse_prop, total_divorced_prop, total_never_married_prop, total_separated_prop,
               total_widowed_prop, total_married_spouse_absent_prop, total_married_af_spouse_prop,
               sample_married_civ_spouse_prop, sample_divorced_prop, sample_never_married_prop,
               sample_separated_prop, sample_widowed_prop, sample_married_spouse_absent_prop,
               sample_married_af_spouse_prop]

    # Only useful for the
    if dataset.lower() == "bank":
        props_x = [total_married_civ_spouse_prop, total_divorced_prop, total_never_married_prop,
                   sample_married_civ_spouse_prop, sample_divorced_prop, sample_never_married_prop]

    print "marital proportions = ", props_x
    return props_x


def compute_employment_proportions(data_set, total_data, sample):
    # remember to remove this hard-coding for all datasets
    if data_set.lower() == "bank":
        fairness_attribute = 1
    else:
        fairness_attribute = 2

    # ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'services',
    #  'student', 'technician', 'unemployed', 'unknown']

    admin_values = ["admin."]
    bc_values = ["blue-collar"]
    entrepreneur_values = ["entrepreneur"]
    housemaid_values = ["housemaid"]
    mgt_values = ["management"]
    retired_values = ["retired"]
    se_values = ["self-employed"]
    services_values = ["services"]
    student_values = ["student"]
    technician_values = ["technician"]
    unemployed_values = ["unemployed"]
    unknown_values = ["unknown"]

    sample_sensitive_attribute_values = evaluate_fairness(total_data, sample, fairness_attribute)

    total_admin_prop = float(
        h.my_counter(pad.compute_proportions(total_data, fairness_attribute), admin_values)) / len(total_data)
    total_bc_prop = float(h.my_counter(pad.compute_proportions(total_data, fairness_attribute), bc_values)) / len(
        total_data)
    total_entrepreneur_prop = float(
        h.my_counter(pad.compute_proportions(total_data, fairness_attribute), entrepreneur_values)) / len(
        total_data)
    total_housemaid_prop = float(
        h.my_counter(pad.compute_proportions(total_data, fairness_attribute), housemaid_values)) / len(
        total_data)
    total_management_prop = float(
        h.my_counter(pad.compute_proportions(total_data, fairness_attribute), mgt_values)) / len(
        total_data)
    total_retired_prop = float(
        h.my_counter(pad.compute_proportions(total_data, fairness_attribute), retired_values)) / len(
        total_data)
    total_se_prop = float(
        h.my_counter(pad.compute_proportions(total_data, fairness_attribute), se_values)) / len(
        total_data)
    total_services_prop = float(
        h.my_counter(pad.compute_proportions(total_data, fairness_attribute), services_values)) / len(
        total_data)
    total_student_prop = float(
        h.my_counter(pad.compute_proportions(total_data, fairness_attribute), student_values)) / len(
        total_data)
    total_technician_prop = float(
        h.my_counter(pad.compute_proportions(total_data, fairness_attribute), technician_values)) / len(
        total_data)
    total_unemployed_prop = float(
        h.my_counter(pad.compute_proportions(total_data, fairness_attribute), unemployed_values)) / len(
        total_data)
    total_unknown_prop = float(
        h.my_counter(pad.compute_proportions(total_data, fairness_attribute), unknown_values)) / len(
        total_data)

    sample_admin_prop = float(h.my_counter(sample_sensitive_attribute_values, admin_values)) / len(sample)
    sample_bc_prop = float(h.my_counter(sample_sensitive_attribute_values, bc_values)) / len(sample)
    sample_entrepreneur_prop = float(h.my_counter(sample_sensitive_attribute_values, entrepreneur_values)) / len(sample)
    sample_housemaid_prop = float(h.my_counter(sample_sensitive_attribute_values, housemaid_values)) / len(sample)
    sample_management_prop = float(h.my_counter(sample_sensitive_attribute_values, mgt_values)) / len(sample)
    sample_retired_prop = float(h.my_counter(sample_sensitive_attribute_values, retired_values)) / len(sample)
    sample_se_prop = float(h.my_counter(sample_sensitive_attribute_values, se_values)) / len(sample)
    sample_services_prop = float(h.my_counter(sample_sensitive_attribute_values, services_values)) / len(sample)
    sample_student_prop = float(h.my_counter(sample_sensitive_attribute_values, student_values)) / len(sample)
    sample_technician_prop = float(h.my_counter(sample_sensitive_attribute_values, technician_values)) / len(sample)
    sample_unemployed_prop = float(h.my_counter(sample_sensitive_attribute_values, unemployed_values)) / len(sample)
    sample_unknown_prop = float(h.my_counter(sample_sensitive_attribute_values, unknown_values)) / len(sample)

    # ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'services',
    #  'student', 'technician', 'unemployed', 'unknown']


    props_x = [total_admin_prop, total_bc_prop, total_entrepreneur_prop, total_housemaid_prop, total_management_prop,
               total_retired_prop, total_se_prop, total_services_prop, total_student_prop, total_technician_prop,
               total_unemployed_prop, total_unknown_prop, sample_admin_prop, sample_bc_prop, sample_entrepreneur_prop,
               sample_housemaid_prop, sample_management_prop, sample_retired_prop, sample_se_prop, sample_services_prop,
               sample_student_prop, sample_technician_prop, sample_unemployed_prop, sample_unknown_prop]

    print "employment proportions = ", props_x
    return props_x


def build_gender_rects(gender_props, ax, ind, width=0.35):
    men_std = (2, 3)
    men_props = (gender_props[0] * 100, gender_props[2] * 100)
    rects1 = ax.bar(ind, men_props, width, color='r', yerr=men_std)

    women_means = (gender_props[1] * 100, gender_props[3] * 100)
    women_std = (3, 5)
    rects2 = ax.bar(ind + width, women_means, width, color='y', yerr=women_std)
    y = [rects1, rects2]
    rect_keys = (rects1[0], rects2[0])
    rect_values = ('Men', 'Women')
    return y, rect_keys, rect_values


# def build_marital_rects(marital_props, ax, ind, width=0.1):
#     print "marital props = ", marital_props
#     married_civ_spouse_std = (2, 3)
#     married_civ_spouse_props = (marital_props[0] * 100, marital_props[7] * 100)
#     rects1 = ax.bar(ind, married_civ_spouse_props, width, color='r', yerr=married_civ_spouse_std)
#
#     divorced_props = (marital_props[1] * 100, marital_props[8] * 100)
#     divorced_std = (3, 5)
#     rects2 = ax.bar(ind + width, divorced_props, width, color='y', yerr=divorced_std)
#
#     never_married_props = (marital_props[2] * 100, marital_props[9] * 100)
#     never_married_std = (3, 5)
#     rects3 = ax.bar(ind + 2 * width, never_married_props, width, color='g', yerr=never_married_std)
#
#     separated_props = (marital_props[3] * 100, marital_props[10] * 100)
#     separated_std = (3, 5)
#     rects4 = ax.bar(ind + 3 * width, separated_props, width, color='c', yerr=separated_std)
#
#     widowed_props = (marital_props[4] * 100, marital_props[11] * 100)
#     widowed_std = (3, 5)
#     rects5 = ax.bar(ind + 4 * width, widowed_props, width, color='b', yerr=widowed_std)
#
#     married_spouse_absent_props = (marital_props[5] * 100, marital_props[12] * 100)
#     married_spouse_absent_std = (3, 5)
#     rects6 = ax.bar(ind + 5 * width, married_spouse_absent_props, width, color='w', yerr=married_spouse_absent_std)
#
#     married_af_spouse_props = (marital_props[6] * 100, marital_props[13] * 100)
#     married_af_spouse_std = (3, 5)
#     rects7 = ax.bar(ind + 6 * width, married_af_spouse_props, width, color='k', yerr=married_af_spouse_std)
#
#     y = [rects1, rects2, rects3, rects4, rects5, rects6, rects7]
#     rect_keys = (rects1[0], rects2[0], rects3[0], rects4[0], rects5[0], rects6[0], rects7[0])
#     rect_values = ('Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent',
#                    'Married-AF-spouse')
#     rect_values = ('m_civ_spouse', 'divorced', 'never_m', 'separated', 'widowed', 'm_spouse_absent',
#                    'm_af_spouse')
#
#     return y, rect_keys, rect_values


def build_marital_rects(dataset, marital_props, ax, ind, width=0.1):
    print "marital props = ", marital_props
    if dataset == "bank":
        married_std = (2, 3)
        married_props = (marital_props[0] * 100, marital_props[3] * 100)
        rects1 = ax.bar(ind, married_props, width, color='r', yerr=married_std)

        divorced_std = (3, 5)
        divorced_props = (marital_props[1] * 100, marital_props[4] * 100)
        rects2 = ax.bar(ind, divorced_props, width, color='g', yerr=divorced_std)

        single_std = (3, 5)
        single_props = (marital_props[2] * 100, marital_props[5] * 100)
        rects3 = ax.bar(ind, single_props, width, color='b', yerr=single_std)

        y = [rects1, rects2, rects3]
        rect_keys = (rects1[0], rects2[0], rects3[0])
        rect_values = ("married", "divorced", "single")

    else:
        married_civ_spouse_std = (2, 3)
        married_civ_spouse_props = (marital_props[0] * 100, marital_props[7] * 100)
        rects1 = ax.bar(ind, married_civ_spouse_props, width, color='r', yerr=married_civ_spouse_std)

        divorced_props = (marital_props[1] * 100, marital_props[8] * 100)
        divorced_std = (3, 5)
        rects2 = ax.bar(ind + width, divorced_props, width, color='y', yerr=divorced_std)

        never_married_props = (marital_props[2] * 100, marital_props[9] * 100)
        never_married_std = (3, 5)
        rects3 = ax.bar(ind + 2 * width, never_married_props, width, color='g', yerr=never_married_std)

        separated_props = (marital_props[3] * 100, marital_props[10] * 100)
        separated_std = (3, 5)
        rects4 = ax.bar(ind + 3 * width, separated_props, width, color='c', yerr=separated_std)

        widowed_props = (marital_props[4] * 100, marital_props[11] * 100)
        widowed_std = (3, 5)
        rects5 = ax.bar(ind + 4 * width, widowed_props, width, color='b', yerr=widowed_std)

        married_spouse_absent_props = (marital_props[5] * 100, marital_props[12] * 100)
        married_spouse_absent_std = (3, 5)
        rects6 = ax.bar(ind + 5 * width, married_spouse_absent_props, width, color='w', yerr=married_spouse_absent_std)

        married_af_spouse_props = (marital_props[6] * 100, marital_props[13] * 100)
        married_af_spouse_std = (3, 5)
        rects7 = ax.bar(ind + 6 * width, married_af_spouse_props, width, color='k', yerr=married_af_spouse_std)

        y = [rects1, rects2, rects3, rects4, rects5, rects6, rects7]
        rect_keys = (rects1[0], rects2[0], rects3[0], rects4[0], rects5[0], rects6[0], rects7[0])
        rect_values = (
            'Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent',
            'Married-AF-spouse')
        rect_values = ('m_civ_spouse', 'divorced', 'never_m', 'separated', 'widowed', 'm_spouse_absent',
                       'm_af_spouse')

    return y, rect_keys, rect_values


def build_race_rects(race_props, ax, ind, width=0.15):
    white_std = (2, 3)
    white_props = (race_props[0] * 100, race_props[5] * 100)
    rects1 = ax.bar(ind, white_props, width, color='r', yerr=white_std)

    black_props = (race_props[1] * 100, race_props[6] * 100)
    black_std = (3, 5)
    rects2 = ax.bar(ind + width, black_props, width, color='y', yerr=black_std)

    asian_props = (race_props[2] * 100, race_props[7] * 100)
    asian_std = (3, 5)
    rects3 = ax.bar(ind + 2 * width, asian_props, width, color='g', yerr=asian_std)

    native_props = (race_props[4] * 100, race_props[9] * 100)
    native_std = (3, 5)
    rects4 = ax.bar(ind + 3 * width, native_props, width, color='c', yerr=native_std)

    other_props = (race_props[3] * 100, race_props[8] * 100)
    other_std = (3, 5)
    rects5 = ax.bar(ind + 4 * width, other_props, width, color='b', yerr=other_std)

    y = [rects1, rects2, rects3, rects4, rects5]
    rect_keys = (rects1[0], rects2[0], rects3[0], rects4[0], rects5[0])
    rect_values = ('White', 'Black', 'Asian', 'Native', 'Other')
    return y, rect_keys, rect_values


def autolabel(ax, rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height,
                '%d' % int(height),
                ha='center', va='bottom')


def plot_multi_bars(ax, fig, ind, rects, rects_keys, rects_values, sensitive_a, div_attribute, k_perc, no_records,
                    fig_name_path, width=0.1):
    width = width  # the width of the bars

    # add some text for labels, title and axes ticks
    ax.set_ylabel('Proportions')
    ax.set_title('Percentage ' + sensitive_a + ' proportions by data groups - diversifying on ' + div_attribute)
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(('All data', 'GMM Sample'))
    ax.set_ylim(0, 100)

    ax.legend(rects_keys, rects_values)

    for rect in rects:
        autolabel(ax, rect)
    # fig.savefig(fig_name_path)
    plt.show()


def sample_details(sample, d):
    sampled = []

    for index in sample:
        sampled.append(d[index])
    return sampled


def describe_sample(sampled, attribute_index):
    values = []

    for item in sampled:
        values.append(item[attribute_index])
    return values


def sample_to_list_of_lists(sample_list):
    g = []
    for item in sample_list:
        g.append([item])
    return g


def compute_div_score(algorithm, sample):
    if algorithm == "max_min":
        sample_max_val = np.max(sample)
        # d = squareform(pdist(sample, metric="euclidean")) / sample_max_val
        d = squareform(pdist(sample, metric="euclidean"))
        np.fill_diagonal(d, np.nan)

        # and calculate the minimum of each row (or column)
        min_distances = np.nanmin(d, axis=1)
        return min(min_distances)
    elif algorithm == "max_sum":
        sample_max_val = np.max(sample)
        d = squareform(pdist(sample, metric="euclidean")) / sample_max_val
        # d = squareform(pdist(sample, metric="euclidean"))
        print "normalized distance matrix for MAXSUM = ", d
        sums = np.sum(d)
        print "sums contains = ", sums
        return 0.5 * sums
    else:
        return None


def obtain_sampled_values(all_data, data_sample, variable_name):
    # Remember to fix this!!!
    if variable_name == "year_weeks":
        variable_index = 38
    elif variable_name == "age":
        variable_index = 0
    elif variable_name == "balance":
        variable_index = 5
    elif variable_name == "capital_gain":
        variable_index = 9
    elif variable_name == "marital_num":
        variable_index = 2
    else:
        variable_index = None

    sampled_records = sample_details(data_sample, all_data)
    # print "sampled records = ", sampled_records

    sampled_values = describe_sample(sampled_records, variable_index)
    # print "sampled values = ", sampled_values

    return sampled_values


def run(k_perc, no_of_records, dataset_name, con_diversification_a, cat_diversification_a, sensitive_a, weight_1,
        weight_2, normalization_method, algorithm, fig_save_path):
    all_data, data_sample, con_div_attribute, cat_div_attribute = sample_on(k_perc, dataset_name, no_of_records,
                                                                            con_diversification_a,
                                                                            cat_diversification_a, weight_1,
                                                                            weight_2, normalization_method, algorithm)
    # details for plotting
    N = 2  # number of groups
    ind = np.arange(N)  # the x locations for the groups
    figure, axes = plt.subplots()

    if sensitive_a.lower() == "gender_2":
        totals = compute_gender_proportions(dataset_name, all_data, data_sample)
        rects, rect_keys, rect_values = build_gender_rects(totals, axes, ind)

    elif sensitive_a.lower() == "race":
        totals = compute_racial_proportions(dataset_name, all_data, data_sample)
        rects, rect_keys, rect_values = build_race_rects(totals, axes, ind)

    elif sensitive_a.lower() == "marital_status":
        totals = compute_marital_proportions(dataset_name, all_data, data_sample)
        rects, rect_keys, rect_values = build_marital_rects(dataset_name, totals, axes, ind)

    else:
        totals = None
        rects = None
        rect_keys = None
        rect_values = None

    sampled_values = obtain_sampled_values(all_data, data_sample, con_diversification_a)
    diversity_score = compute_div_score(algorithm, np.asmatrix(np.array(sample_to_list_of_lists(sampled_values))))
    print "diversity score = ", diversity_score

    # plot
    if totals is not None and rects is not None and rect_keys is not None and rect_values is not None:
        plot_multi_bars(axes, figure, ind, rects, rect_keys, rect_values, sensitive_a, con_diversification_a, k_perc,
                        no_of_records, fig_save_path)

    return totals


def run2(xs, k_perc, no_of_records, dataset_name, con_diversification_a, cat_diversification_a, sensitive_a,
         normalization_method, algorithm):
    gmm_alls = []
    con_diversity_scores = []
    cat_diversity_scores = []

    for i in xs:
        weight_1 = float(i)
        weight_2 = 1 - weight_1

        all_data, data_sample, con_div_attribute, cat_div_attribute = sample_on(k_perc, dataset_name, no_of_records,
                                                                                con_diversification_a,
                                                                                cat_diversification_a, weight_1,
                                                                                weight_2,
                                                                                normalization_method, algorithm)

        if sensitive_a.lower() == "gender":
            totals = compute_gender_proportions(dataset_name, all_data, data_sample)
            # rects, rect_keys, rect_values = build_gender_rects(totals, axes, ind)

        elif sensitive_a.lower() == "race":
            totals = compute_racial_proportions(dataset_name, all_data, data_sample)
            # rects, rect_keys, rect_values = build_race_rects(totals, axes, ind)

        elif sensitive_a.lower() == "marital_status":
            totals = compute_marital_proportions(dataset_name, all_data, data_sample)
            # rects, rect_keys, rect_values = build_marital_rects(totals, axes, ind)
        elif sensitive_a.lower() == "employment":
            totals = compute_employment_proportions(dataset_name, all_data, data_sample)
        else:
            totals = None
        start = (len(totals) / 2)
        gmm_totals = totals[start:]
        gmm_alls.append(gmm_totals)

        sampled_con_values = obtain_sampled_values(all_data, data_sample, con_diversification_a)

        if cat_diversification_a == "marital_num":
            sampled_cat_str_values = obtain_sampled_values(all_data, data_sample, cat_diversification_a)
            sampled_cat_values = []
            for j in sampled_cat_str_values:
                sampled_cat_values.append(pbd.marital_status_number(j))
        else:
            sampled_cat_values = obtain_sampled_values(all_data, data_sample, cat_diversification_a)

        con_diversity_score = compute_div_score(algorithm,
                                                np.asmatrix(np.array(sample_to_list_of_lists(sampled_con_values))))
        cat_diversity_score = compute_div_score(algorithm,
                                                np.asmatrix(np.array(sample_to_list_of_lists(sampled_cat_values))))

        print "con diversity score = ", con_diversity_score
        print "cat diversity score = ", cat_diversity_score
        print "balances chosen = ", sampled_con_values
        print "marital status chosen = ", sampled_cat_values

        con_diversity_scores.append(con_diversity_score)
        cat_diversity_scores.append(cat_diversity_score)

        print "w1= ", str(weight_1), " w2= ", str(weight_2), " totals = ", totals
    print "all gmm totals = ", gmm_alls
    print "con diversity scores = ", con_diversity_scores
    print "cat diversity scores = ", cat_diversity_scores

    return xs, gmm_alls, con_diversity_scores, cat_diversity_scores


def plot_with_line(xis, k_perc, no_of_records, dataset_name, con_diversification_a, cat_diversification_a, sensitive_a,
                   normalization_method, algorithm, fig_save_path):
    x, totals, con_div_scores, cat_div_scores = run2(xis, k_perc, no_of_records, dataset_name, con_diversification_a,
                                                     cat_diversification_a, sensitive_a, normalization_method,
                                                     algorithm)

    # x = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    # x = [0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1]
    # x = [0.99, 0.991, 0.992, 0.993, 0.994, 0.995, 0.996, 0.997, 0.998, 0.999, 1]

    title = algorithm + ": diversity scores, diversifying on " + con_diversification_a + " and " + cat_diversification_a

    base_path = "./data/bank_marketing/results/diversity_scores/"

    save_path = base_path + algorithm + "/" + con_diversification_a + "_" + cat_diversification_a + "_N=" + str(
        no_of_records) + "_k=" + str(k_perc) + "_dataset=" + dataset_name + "_min_weight_1=" + str(x[0]) \
                + "_diversity_scores.png"

    plot_diversity_score_lines(x, con_div_scores, cat_div_scores, con_diversification_a, cat_diversification_a,
                               title, save_path)

    ind = np.arange(11)
    figure, axes = plt.subplots()

    avg_bar1 = []
    avg_bar2 = []
    avg_bar3 = []

    for i in totals:
        avg_bar1.append(i[0] * 100)
        avg_bar2.append(i[1] * 100)
        avg_bar3.append(i[2] * 100)

    # rects1 = plt.bar(ind, avg_bar1, 0.15, color='#ff0000', label='male')
    # rects2 = plt.bar(ind + 0.15, avg_bar2, 0.15, color='#00ff00', label='female')

    ## marital_status_bank
    rects1 = plt.bar(ind, avg_bar1, 0.20, color='#ff0000', label='married')
    rects2 = plt.bar(ind + 0.20, avg_bar2, 0.20, color='#00ff00', label='divorced')
    rects3 = plt.bar(ind + 0.40, avg_bar3, 0.20, color='blue', label='single')

    ## race
    # rects1 = plt.bar(ind, avg_bar1, 0.20, color='#ff0000', label='white')
    # rects2 = plt.bar(ind + 0.20, avg_bar2, 0.20, color='#00ff00', label='black')
    # rects3 = plt.bar(ind + 0.40, avg_bar3, 0.20, color='blue', label='asian')
    # rects4 = plt.bar(ind + 0.60, avg_bar4, 0.20, color='cyan', label='native')
    # rects5 = plt.bar(ind + 0.80, avg_bar5, 0.20, color='grey', label='other')

    ## marital_status other

    # rects1 = plt.bar(ind, avg_bar1, 0.20, color='#ff0000', label='married-civ')
    # rects2 = plt.bar(ind + 0.20, avg_bar2, 0.20, color='#00ff00', label='divorced')
    # rects3 = plt.bar(ind + 0.40, avg_bar3, 0.20, color='blue', label='never-married')
    # rects4 = plt.bar(ind + 0.60, avg_bar4, 0.20, color='cyan', label='separated')
    # rects5 = plt.bar(ind + 0.80, avg_bar5, 0.20, color='grey', label='widowed')
    # rects6 = plt.bar(ind + 0.10, avg_bar5, 0.20, color='yellow', label='married-sa')
    # rects7 = plt.bar(ind + 0.12, avg_bar5, 0.20, color='black', label='married-af')

    # ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'services',
    #  'student', 'technician', 'unemployed', 'unknown']

    ## banking employment options
    # rects1 = plt.bar(ind, avg_bar1, 0.20, color='#ff0000', label='admin')
    # rects2 = plt.bar(ind + 0.20, avg_bar2, 0.20, color='#00ff00', label='blue-collar')
    # rects3 = plt.bar(ind + 0.40, avg_bar3, 0.20, color='blue', label='entrepreneur')
    # rects4 = plt.bar(ind + 0.60, avg_bar4, 0.20, color='cyan', label='housemaid')
    # rects5 = plt.bar(ind + 0.80, avg_bar5, 0.20, color='grey', label='management')
    # rects6 = plt.bar(ind + 0.10, avg_bar5, 0.20, color='yellow', label='retired')
    # rects7 = plt.bar(ind + 0.12, avg_bar5, 0.20, color='black', label='self-employed')

    # high_point_x = []
    # high_point_y = []
    # for i in range(0, 11):
    #     single_bar_group = {rects1[i].get_height(): rects1[i].get_x() + rects1[i].get_width() / 2.0,
    #                         rects2[i].get_height(): rects2[i].get_x() + rects2[i].get_width() / 2.0,
    #                         rects3[i].get_height(): rects3[i].get_x() + rects3[i].get_width() / 2.0}
    #     height_list = list(single_bar_group.keys())
    #     height_list.sort(reverse=True)
    #     for single_height in height_list:
    #         # high_point_y.append(single_height)
    #         high_point_x.append(single_bar_group[single_height])
    #         break

    # for i in range(0, len(div_scores)):
    #     # multiplying div_score by 10 so it's visible on the chart
    #     # high_point_y.append(i * 10)
    #     high_point_y.append(div_scores[i] * 1)
    #     # high_point_y.append(float(div_scores[i])/float(div_scores[0]) * 1)

    # print "high_point_y = ", high_point_y

    # trend_line = plt.plot(high_point_x, high_point_y, marker='o', color='#5b74a8', label='Trend Line')

    # rects = [rects1, rects2]
    rects = [rects1, rects2, rects3]
    # rects = [rects1, rects2, rects3, rects4, rects5, rects6, rects7]

    for rect in rects:
        autolabel(axes, rect)

    plt.xlabel("weight of attribute " + "'" + con_diversification_a + "'")
    plt.ylabel('sensitive attribute representation (%)')
    plt.title(algorithm + ': diversifying on ' + con_diversification_a + " and " + cat_diversification_a + ". N=" + str(
        no_of_records) + ". k= " + str(k_perc))
    plt.xticks(ind + 0.20, x)
    # plt.xticks(ind + 0.15, x2)
    # plt.xticks(ind + 0.15, x3)
    plt.legend()
    # plt.legend()
    # figure.savefig(fig_save_path)
    # plt.show()


def plot_single_bars(k_perc, no_of_records, dataset_name, con_diversification_a, cat_diversification_a, sensitive_a,
                     weight_1,
                     weight_2, normalization_method, algorithm, fig_save_path):
    plt.rcdefaults()
    # objects = ('Python', 'C++', 'Java', 'Perl', 'Scala', 'Lisp')
    objects = ('Married', 'Divorced', 'Single')
    x_pos = np.arange(len(objects))
    performance = [10, 8, 6, 4, 2, 1]

    totals = run(k_perc, no_of_records, dataset_name, con_diversification_a, cat_diversification_a, sensitive_a,
                 weight_1, weight_2, normalization_method, algorithm, fig_save_path)

    print totals

    # plt.bar(y_pos, performance, align='center', alpha=0.5)
    bar_list = plt.bar(x_pos, totals[0:3], align='center', color='green', alpha=0.5)
    bar_list[0].set_color('r')
    bar_list[1].set_color('g')
    bar_list[2].set_color('b')
    plt.xticks(x_pos, objects)
    plt.ylabel('proportion in data (%)')
    plt.title('data proportions by ' + sens_a)
    plt.savefig(fig_save_path)

    plt.show()


def plot_diversity_score_lines(x, con_diversity_scores, cat_diversity_scores, con_attribute_name, cat_attribute_name,
                               title, save_path):
    # x = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    plt.plot(x, con_diversity_scores, 'r*-', label=con_attribute_name)
    plt.plot(x, cat_diversity_scores, 'go-', label=cat_attribute_name)

    plt.xlabel("weight of '" + con_attribute_name + "'")
    plt.ylabel("diversity score")

    plt.title(title)

    plt.legend()

    plt.show()
    plt.savefig(save_path)


perc = 10
records = 4522
# dataset = "census"
dataset = "bank"
# dataset = "adult"
# div_con_a = "year_weeks"
# div_con_a = "age"
div_con_a = "balance"
# div_con_a = "capital_gain"
# div_cat_a = "gender_num"
# div_cat_a = "marital_status_num"
# div_cat_a = "marital_num"  # marital_num
div_cat_a = "age"  # marital_num
# div_cat_a = "job_num" #marital_num
# div_cat_a = "race_num"
# sens_a = "gender"
sens_a = "marital_status"
# sens_a = "employment"
# sens_a = "race"

# algorithm = "max_min"
algorithm = "max_sum"

# normalize_by = "max_val"
normalize_by = "max_diff"

xs = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
# xs = [0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1]
# xs = [0.99, 0.991, 0.992, 0.993, 0.994, 0.995, 0.996, 0.997, 0.998, 0.999, 1]

w1 = 1
w2 = 1

# path = "./data/kdd_census/results/multi_variable/weights/"+sens_a+"/"+algorithm+"/"
path = "./data/bank_marketing/results/" + sens_a + "/" + algorithm + "/"
path1 = "./data/bank_marketing/results/general/bank_dataset_marital_status_proportions.png"

name = div_con_a + "_" + sens_a + "_N=" + str(records) + "_k=" + str(perc) + "_dataset=" + dataset + "_enlarged"

save_fig_at = path + div_con_a + "_" + sens_a + "_N=" + str(records) + "_k=" + str(
    perc) + "_dataset=" + dataset + ".png"

print "file name = ", name

# run(perc, records, dataset, div_con_a, div_cat_a, sens_a, w1, w2, normalize_by, algorithm, save_fig_at)
# run2(perc, records, dataset, div_con_a, div_cat_a, sens_a, normalize_by, algorithm)
# plot_with_line(xs, perc, records, dataset, div_con_a, div_cat_a, sens_a, normalize_by, algorithm, save_fig_at)
plot_single_bars(perc, records, dataset, div_con_a, div_cat_a, sens_a, w1, w2, normalize_by, algorithm, path1)
