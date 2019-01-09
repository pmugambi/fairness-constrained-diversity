import numpy as np
import prepare_adult_data as pad
import gmm as gmm
import math
import helpers as h
import matplotlib.pyplot as plt


def sample_diverse_k(data, k=10):
    # create a numpy matrix of the data
    d = np.array(data)
    print "length of data received = ", len(data)

    print "d = ", d

    # normalize columns
    d_normed = d / d.max(axis=0)
    print d_normed

    # get 5% of the data
    k = math.ceil(0.05 * len(data))

    print "k = ", k

    top_k = gmm.greedy_diverse(d_normed, k)
    print top_k

    return top_k


def evaluate_fairness(data, sample, fairness_attributes):
    sample_sensitive_attributes_values = []
    for index in sample:
        data_line = data[index]
        # for fa in fairness_atributes:
        sample_fairness_attribute_value = data_line[fairness_attributes[0]]
        sample_sensitive_attributes_values.append(sample_fairness_attribute_value)

    print sample_sensitive_attributes_values
    return sample_sensitive_attributes_values


def sample_on(x, diversification_attribute):
    total_data = pad.process(x)
    data = pad.process(x, [diversification_attribute])  # obtain data based on diversification attribute
    sample = sample_diverse_k(data)

    return total_data, sample, diversification_attribute


def compute_gender_proportions(total_data, sample):
    fairness_attributes = pad.obtain_sensitive_attributes_columns(["gender"])
    sample_sensitive_attributes_values = evaluate_fairness(total_data, sample, fairness_attributes)

    total_female_prop = float(
        h.my_counter(pad.compute_proportions(total_data, fairness_attributes[0]), "female")) / len(total_data)
    total_male_prop = float(h.my_counter(pad.compute_proportions(total_data, fairness_attributes[0]), "male")) / len(
        total_data)

    sample_female_prop = float(h.my_counter(sample_sensitive_attributes_values, "female")) / len(sample)
    sample_male_prop = float(h.my_counter(sample_sensitive_attributes_values, "male")) / len(sample)

    props_x = [total_male_prop, total_female_prop, sample_male_prop, sample_female_prop]

    print "proportions = ", props_x
    return props_x


def compute_racial_proportions(total_data, sample):
    fairness_attributes = pad.obtain_sensitive_attributes_columns(["race"])
    sample_sensitive_attributes_values = evaluate_fairness(total_data, sample, fairness_attributes)

    total_white_prop = float(
        h.my_counter(pad.compute_proportions(total_data, fairness_attributes[0]), "white")) / len(total_data)
    total_black_prop = float(h.my_counter(pad.compute_proportions(total_data, fairness_attributes[0]), "black")) / len(
        total_data)
    total_asian_prop = float(h.my_counter(pad.compute_proportions(total_data, fairness_attributes[0]), "asian-pac-islander")) / len(
        total_data)
    total_other_prop = float(h.my_counter(pad.compute_proportions(total_data, fairness_attributes[0]), "other")) / len(
        total_data)
    total_native_a_prop = float(h.my_counter(pad.compute_proportions(total_data, fairness_attributes[0]), "amer-indian-eskimo")) / len(
        total_data)

    sample_white_prop = float(h.my_counter(sample_sensitive_attributes_values, "white")) / len(sample)
    sample_black_prop = float(h.my_counter(sample_sensitive_attributes_values, "black")) / len(sample)
    sample_asian_prop = float(h.my_counter(sample_sensitive_attributes_values, "asian-pac-islander")) / len(sample)
    sample_other_prop = float(h.my_counter(sample_sensitive_attributes_values, "other")) / len(sample)
    sample_native_a_prop = float(h.my_counter(sample_sensitive_attributes_values, "amer-indian-eskimo")) / len(sample)

    props_x = [total_white_prop, total_black_prop, total_asian_prop, total_other_prop, total_native_a_prop,
               sample_white_prop, sample_black_prop, sample_asian_prop, sample_other_prop, sample_native_a_prop]

    print "proportions = ", props_x
    return props_x


def build_gender_rects(gender_props, ax, ind, width=0.2):
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


def plot_multi_bars(ax, fig, rects, rects_keys, rects_values, sensitive_a, div_attribute, k_perc, width=0.2):
    print "rects contains = ", rects
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
    # fig.savefig("./data/results/" + div_attribute + "_" + sensitive_a + "_proportions_kperc_"+str(k_perc))
    fig.savefig("./data/results/" + div_attribute + "_" + sensitive_a + "_proportions_ss_100")
    plt.show()

k_perc = 5
all_data, data_sample, div_attribute = sample_on(100, "education-num")
gender_totals = compute_gender_proportions(all_data, data_sample)
# racial_totals = compute_racial_proportions(all_data, data_sample)

# plot

N = 2  # number of groups
ind = np.arange(N)  # the x locations for the groups
figure, axes = plt.subplots()
g_rects, g_rect_keys, g_rect_values = build_gender_rects(gender_totals, axes, ind)
plot_multi_bars(axes, figure, g_rects, g_rect_keys, g_rect_values, "gender", "education-num", k_perc)
# r_rects, r_rect_keys, r_rect_values = build_race_rects(racial_totals, axes, ind)
# plot_multi_bars(axes, figure, r_rects, r_rect_keys, r_rect_values, "race", "age", k_perc)
