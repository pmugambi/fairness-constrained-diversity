import numpy as np
import prepare_adult_data as pad
import prepare_kdd_census_data as pkcd
import gmm as gmm
import gms as gms
import math
import helpers as h
from scipy.spatial.distance import pdist, squareform, euclidean
import matplotlib.pyplot as plt


def compute_highest_difference(data):
    min_v = min(data)[0]
    max_v = max(data)[0]

    return max_v - min_v


def get_highest_value(data):
    return max(data)[0]


def sample_mixed_data_diverse_k(con_data, cat_data, norm_val, k, weight_1, weight_2, algorithm):
    con_d = np.asmatrix(np.array(con_data))
    cat_d = np.asmatrix(np.array(cat_data))
    if algorithm == "maxmin":
        top_k = gmm.greedy_diverse_mod(con_d, cat_d, norm_val, k, weight_1, weight_2)
    elif algorithm == "maxsum":
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

    print "gender proportions = ", props_x
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
    else:
        fairness_attribute = pkcd.obtain_sensitive_attribute_column("marital_status")[0]

    nevermarried_values = ["never-married", "never married"]
    marriedciv_values = ["married-civilian spouse present", "married-civ-spouse"]
    marriedabsentspouse_values = ["married-spouse absent", "married-spouse-absent"]
    separated_values = ["separated"]
    divorced_values = ["divorced"]
    widowed_values = ["widowed"]
    marriedaf_values = ["married-a f spouse present", "married-af-spouse"]

    sample_sensitive_attribute_values = evaluate_fairness(total_data, sample, fairness_attribute)

    total_married_civ_spouse_prop = float(
        h.my_counter(pad.compute_proportions(total_data, fairness_attribute), marriedciv_values)) / len(total_data)
    total_divorced_prop = float(h.my_counter(pad.compute_proportions(total_data, fairness_attribute), divorced_values)) / len(
        total_data)
    total_never_married_prop = float(
        h.my_counter(pad.compute_proportions(total_data, fairness_attribute), nevermarried_values)) / len(
        total_data)
    total_separated_prop = float(h.my_counter(pad.compute_proportions(total_data, fairness_attribute), separated_values)) / len(
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

    sample_married_civ_spouse_prop = float(h.my_counter(sample_sensitive_attribute_values, marriedciv_values)) / len(sample)
    sample_divorced_prop = float(h.my_counter(sample_sensitive_attribute_values, divorced_values)) / len(sample)
    sample_never_married_prop = float(h.my_counter(sample_sensitive_attribute_values, nevermarried_values)) / len(sample)
    sample_separated_prop = float(h.my_counter(sample_sensitive_attribute_values, separated_values)) / len(sample)
    sample_widowed_prop = float(h.my_counter(sample_sensitive_attribute_values, widowed_values)) / len(sample)
    sample_married_spouse_absent_prop = float(h.my_counter(sample_sensitive_attribute_values, marriedabsentspouse_values)) / len(sample)
    sample_married_af_spouse_prop = float(h.my_counter(sample_sensitive_attribute_values, marriedaf_values)) / len(sample)

    props_x = [total_married_civ_spouse_prop, total_divorced_prop, total_never_married_prop, total_separated_prop,
               total_widowed_prop, total_married_spouse_absent_prop, total_married_af_spouse_prop,
               sample_married_civ_spouse_prop, sample_divorced_prop, sample_never_married_prop,
               sample_separated_prop, sample_widowed_prop, sample_married_spouse_absent_prop,
               sample_married_af_spouse_prop]

    print "marital proportions = ", props_x
    return props_x


def build_gender_rects(gender_props, ax, ind, width=0.35):
    men_std = (2, 3)
    men_props = (gender_props[0]*100, gender_props[2]*100)
    rects1 = ax.bar(ind, men_props, width, color='r', yerr=men_std)

    women_means = (gender_props[1]*100, gender_props[3]*100)
    women_std = (3, 5)
    rects2 = ax.bar(ind + width, women_means, width, color='y', yerr=women_std)
    y = [rects1, rects2]
    rect_keys = (rects1[0], rects2[0])
    rect_values = ('Men', 'Women')
    return y, rect_keys, rect_values


def build_marital_rects(marital_props, ax, ind, width=0.1):
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
    rect_values = ('Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent',
                   'Married-AF-spouse')
    rect_values = ('m_civ_spouse', 'divorced', 'never_m', 'separated', 'widowed', 'm_spouse_absent',
                   'm_af_spouse')

    return y, rect_keys, rect_values


def build_race_rects(race_props, ax, ind, width=0.15):
    white_std = (2, 3)
    white_props = (race_props[0] * 100, race_props[5] * 100)
    rects1 = ax.bar(ind, white_props, width, color='r', yerr=white_std)

    black_props = (race_props[1]*100, race_props[6]*100)
    black_std = (3, 5)
    rects2 = ax.bar(ind + width, black_props, width, color='y', yerr=black_std)

    asian_props = (race_props[2]*100, race_props[7]*100)
    asian_std = (3, 5)
    rects3 = ax.bar(ind + 2*width, asian_props, width, color='g', yerr=asian_std)

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
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')


def plot_multi_bars(ax, fig, ind, rects, rects_keys, rects_values, sensitive_a, div_attribute, k_perc, no_records, fig_name_path, width=0.1):
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


def compute_div_score(sample):
    d = squareform(pdist(sample, metric="euclidean"))
    np.fill_diagonal(d, np.nan)
    # print "dist mat = ", d

    # and calculate the minimum of each row (or column)
    min_distances = np.nanmin(d, axis=1)
    return min(min_distances)


def obtain_sampled_values(all_data, data_sample, variable_name):
    # Remember to fix this!!!
    if variable_name == "year_weeks":
        variable_index = 38
    elif variable_name == "age":
        variable_index = 0
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

    if sensitive_a.lower() == "gender":
        totals = compute_gender_proportions(dataset_name, all_data, data_sample)
        rects, rect_keys, rect_values = build_gender_rects(totals, axes, ind)

    elif sensitive_a.lower() == "race":
        totals = compute_racial_proportions(dataset_name, all_data, data_sample)
        rects, rect_keys, rect_values = build_race_rects(totals, axes, ind)

    elif sensitive_a.lower() == "marital_status":
        totals = compute_marital_proportions(dataset_name, all_data, data_sample)
        rects, rect_keys, rect_values = build_marital_rects(totals, axes, ind)

    else:
        totals = None
        rects = None
        rect_keys = None
        rect_values = None

    sampled_values = obtain_sampled_values(all_data, data_sample, con_diversification_a)
    diversity_score = compute_div_score(np.asmatrix(np.array(sample_to_list_of_lists(sampled_values))))
    print "diversity score = ", diversity_score

    # plot
    if totals is not None and rects is not None and rect_keys is not None and rect_values is not None:
        plot_multi_bars(axes, figure, ind, rects, rect_keys, rect_values, sensitive_a, con_diversification_a, k_perc,
                        no_of_records, fig_save_path)


def run2(k_perc, no_of_records, dataset_name, con_diversification_a, cat_diversification_a, sensitive_a,
         normalization_method, algorithm):
    alls = []
    diversity_scores = []
    for i in xrange(0, 11):
        weight_1 = float(i)/float(10)
        weight_2 = 1 - w1
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
        else:
            totals = None
        start = (len(totals)/2)
        gmm_totals = totals[start:]
        alls.append(gmm_totals)

        sampled_values = obtain_sampled_values(all_data, data_sample, con_diversification_a)
        diversity_score = compute_div_score(np.asmatrix(np.array(sample_to_list_of_lists(sampled_values))))
        print "diversity score = ", diversity_score

        diversity_scores.append(diversity_score)

        print "w1= ", str(w1), " w2= ", str(w2), " totals = ", totals
    print "all totals = ", alls
    print "diversity scores = ", diversity_scores

    return alls, diversity_scores


def plot_with_line(k_perc, no_of_records, dataset_name, diversification_a, sensitive_a, normalization_method,
                   algorithm, fig_save_path):
    ind = np.arange(11)
    figure, axes = plt.subplots()
    avg_bar1 = (81191, 79318, 57965, 60557, 14793, 14793, 14793, 14793, 14793, 14793, 14793)
    avg_bar2 = (26826, 26615, 31364, 31088, 55472, 31088, 31088, 31088, 31088, 31088, 31088)
    # avg_bar3 = (36232, 38038, 38615, 39014, 40812)
    # avg_bar4 = (26115, 25879, 25887, 28326, 27988)

    avg_bar1 = []
    avg_bar2 = []
    avg_bar3 = []
    avg_bar4 = []
    avg_bar5 = []

    totals, div_scores = run2(k_perc, no_of_records, dataset_name, diversification_a, sensitive_a,
                              normalization_method, algorithm)

    for i in totals:
        avg_bar1.append(i[0] * 100)
        avg_bar2.append(i[1] * 100)
        avg_bar3.append(i[2] * 100)
        avg_bar4.append(i[3] * 100)
        avg_bar5.append(i[4] * 100)

    # rects1 = plt.bar(ind, avg_bar1, 0.15, color='#ff0000', label='male')
    rects1 = plt.bar(ind, avg_bar1, 0.20, color='#ff0000', label='white')
    # rects2 = plt.bar(ind + 0.15, avg_bar2, 0.15, color='#00ff00', label='female')
    rects2 = plt.bar(ind + 0.20, avg_bar2, 0.20, color='#00ff00', label='black')
    rects3 = plt.bar(ind + 0.40, avg_bar3, 0.20, color='blue', label='asian')
    rects4 = plt.bar(ind + 0.60, avg_bar4, 0.20, color='cyan', label='native')
    rects5 = plt.bar(ind + 0.80, avg_bar5, 0.20, color='grey', label='other')

    high_point_x = []
    high_point_y = []
    for i in range(0, 11):
        single_bar_group = {rects1[i].get_height(): rects1[i].get_x() + rects1[i].get_width() / 2.0,
                            rects2[i].get_height(): rects2[i].get_x() + rects2[i].get_width() / 2.0,
                            rects3[i].get_height(): rects3[i].get_x() + rects3[i].get_width() / 2.0,
                            rects4[i].get_height(): rects4[i].get_x() + rects4[i].get_width() / 2.0,
                            rects5[i].get_height(): rects5[i].get_x() + rects5[i].get_width() / 2.0}

        height_list = list(single_bar_group.keys())
        height_list.sort(reverse=True)
        for single_height in height_list:
            # high_point_y.append(single_height)
            high_point_x.append(single_bar_group[single_height])
            break

    for i in div_scores:
        # multiplying div_score by 10 so it's visible on the chart
        high_point_y.append(i * 10)

    trend_line = plt.plot(high_point_x, high_point_y, marker='o', color='#5b74a8', label='Trend Line')

    # rects = [rects1, rects2]
    rects = [rects1, rects2, rects3, rects4, rects5]

    for rect in rects:
        autolabel(axes, rect)

    plt.xlabel('weights')
    plt.ylabel('sensitive attribute representation (%)')
    plt.xticks(ind + 0.15, ('0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1'))
    plt.legend()
    # figure.savefig(fig_save_path)
    plt.show()


perc = 50
records = 20000
dataset = "census"
div_con_a = "year_weeks"
# div_con_a = "age"
div_cat_a = "gender_num"
# div_cat_a = "race_num"
sens_a = "gender"
# sens_a = "race"

# algorithm = "maxmin"
algorithm = "maxsum"

# normalize_by = "max_val"
normalize_by = "max_diff"

w1 = 1
w2 = 1

save_fig_at = ""

run(perc, records, dataset, div_con_a, div_cat_a, sens_a, w1, w2, normalize_by, algorithm, save_fig_at)
# plot_with_line(perc, records, dataset, div_a, sens_a)
# run2(perc, records, dataset, div_a, sens_a)
