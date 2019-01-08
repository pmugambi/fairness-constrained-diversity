import numpy as np
import prepare_adult_data as pad
import gmm as gmm
import math
import helpers as h
import matplotlib.pyplot as plt


def sample_diverse_k(data, k):
    # create a numpy matrix of the data
    d = np.array(data)
    print "length of data received = ", len(data)

    # normalize columns
    d_normed = d / d.max(axis=0)
    print d_normed

    # get 10% of the data
    k = math.ceil(0.3 * len(data))

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

# x = [100, 1000, 3000, 10000]
x = [100]

totals = []
for i in x:
    # props_x = []
    total_data = pad.process(i)
    data = pad.process(i, ["age"])
    sample = sample_diverse_k(data, 300)
    fairness_attributes = pad.obtain_sensitive_attributes_columns(["gender"])
    # print "fairness attributes = ", fairness_attributes
    sample_sensitive_attributes_values = evaluate_fairness(total_data, sample, fairness_attributes)

    # print "checking this ", h.my_counter(pad.compute_proportions(total_data, fairness_attributes[0]), "female")
    # print "checking this 2 ", h.my_counter(pad.compute_proportions(total_data, fairness_attributes[0]), "male")

    total_female_prop = float(h.my_counter(pad.compute_proportions(total_data, fairness_attributes[0]), "female")) / len(total_data)
    total_male_prop = float(h.my_counter(pad.compute_proportions(total_data, fairness_attributes[0]), "male")) / len(total_data)

    sample_female_prop = float(h.my_counter(sample_sensitive_attributes_values, "female")) / len(sample)
    sample_male_prop = float(h.my_counter(sample_sensitive_attributes_values, "male")) / len(sample)

    # print total_male_prop, total_female_prop, sample_male_prop, sample_female_prop
    props_x = [total_male_prop, total_female_prop, sample_male_prop, sample_female_prop]

    print "proportions = ", props_x

    totals = props_x

print totals

# plot
# import numpy as np
# import matplotlib.pyplot as plt

# N = 5
N = 2
# men_means = (20, 35, 30, 35, 27)
men_means = (totals[0]*100, totals[2]*100)
# men_std = (2, 3, 4, 1, 2)
men_std = (2, 1)
# men_std = (1, 1)

ind = np.arange(N)  # the x locations for the groups
width = 0.25      # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind, men_means, width, color='r', yerr=men_std)

# women_means = (25, 32, 34, 20, 25)
women_means = (totals[1]*100, totals[3]*100)
# women_std = (3, 5, 2, 3, 3)
women_std = (3, 5)
rects2 = ax.bar(ind + width, women_means, width, color='y', yerr=women_std)

# add some text for labels, title and axes ticks
ax.set_ylabel('Proportions')
ax.set_title('Percentage gender proportions by data groups')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(('All data', 'GMM Sample'))
ax.set_ylim(0, 100)
# ax.set_xticklabels(('All data', 'GMM Sample', 'G3', 'G4', 'G5'))

ax.legend((rects1[0], rects2[0]), ('Men', 'Women'))


def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

fig.savefig("./data/results/gender_proportions")

plt.show()


# def plot_multi_bars(group_names, group_data)