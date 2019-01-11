from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import helpers
import prepare_adult_data as pad


# x: diversification attribute, y: sensitive attribute
def compute_correlation(x, y):
    return stats.pearsonr(x, y)

# get correlation between age and gender

age = "age"
gender = "gender_num"
fnlwgt = "fnlwgt"
education = "education_num"
workclass = "work_class_num"
maritalstatus = "marital_status_num"
occupation = "occupation_num"
relationship = "relationship_num"
race = "race_num"
capital_gain = "capital_gain"
capital_loss = "capital_loss"
hours_per_week = "hours_per_week"
native_country = "native_country_num"


ages = pad.process(100, [age])
genders = pad.process(100, [gender])
fnlwgts = pad.process(100, [fnlwgt])
educations = pad.process(100, [education])
workclasses = pad.process(100, [workclass])
maritalstatuses = pad.process(100, [maritalstatus])
occupations = pad.process(100, [occupation])
relationships = pad.process(100, [relationship])
races = pad.process(100, [race])
capital_gains = pad.process(100, [capital_gain])
capital_losses = pad.process(100, [capital_loss])
hours_per_weeks = pad.process(100, [hours_per_week])
native_countries = pad.process(100, [native_country])


ages_list = helpers.list_of_lists_to_list(ages)
genders_list = helpers.list_of_lists_to_list(genders)
fnlwgts_list = helpers.list_of_lists_to_list(fnlwgts)
educations_list = helpers.list_of_lists_to_list(educations)
workclasses_list = helpers.list_of_lists_to_list(workclasses)
maritalstatuses_list = helpers.list_of_lists_to_list(maritalstatuses)
occupations_list = helpers.list_of_lists_to_list(occupations)
relationships_list = helpers.list_of_lists_to_list(relationships)
races_list = helpers.list_of_lists_to_list(races)
capital_gains_list = helpers.list_of_lists_to_list(capital_gains)
capital_losses_list = helpers.list_of_lists_to_list(capital_losses)
hours_per_weeks_list = helpers.list_of_lists_to_list(hours_per_weeks)
native_countries_list = helpers.list_of_lists_to_list(native_countries)




#
# print "age vs gender = ", compute_correlation(ages, genders)
# # # get correlation between fnlwgt and age
# # print "age vs fnlwgt = ", compute_correlation(ages, fnlwgts)
# # # get correlation between fnlwgt and gender
# # print "fnlwgt vs gender = ", compute_correlation(fnlwgts, genders)
# # # get correlation between education and gender
# # print "education vs gender = ", compute_correlation(educations, genders)
# #
# # print stats.pearsonr([1,2,3,4,5], [5,6,7,8,7])
# # print stats.pearsonr([[1], [2], [3], [4], [5]], [[5], [6], [7], [8], [7]])
#
#
# # print np.corrcoef([ages, genders])
#
# # print np.corrcoef([[1,2,3,4,5], [5,6,7,8,7]])
# # print np.corrcoef([[[1], [2], [3], [4], [5]], [[5], [6], [7], [8], [7]]])
#
# b = [[1], [2], [3], [4], [5]]
# print helpers.list_of_lists_to_list(b)
#
#
coefs = np.corrcoef([ages_list, genders_list, fnlwgts_list, educations_list, workclasses_list, maritalstatuses_list,
                   occupations_list, relationships_list, races_list, capital_gains_list, capital_losses_list,
                   hours_per_weeks_list, native_countries_list])

coefs2 = np.copy(coefs)

np.fill_diagonal(coefs2, np.nan)

# print coefs
print coefs.max()
print coefs.min()

min_distances = np.nanmin(coefs2, axis=1)
max_distances = np.nanmax(coefs2, axis=1)

# print min_distances
# print max_distances

print min(min_distances)
print max(max_distances)

print compute_correlation(capital_gains_list, educations_list)
print compute_correlation(capital_losses_list, ages_list)

print np.around(coefs, 3)

# plt.matshow(coefs)
# plt.show()



# vegetables = ["cucumber", "tomato", "lettuce", "asparagus",
#               "potato", "wheat", "barley"]

# farmers = ["Farmer Joe", "Upland Bros.", "Smith Gardening",
#            "Agrifun", "Organiculture", "BioGoods Ltd.", "Cornylee Corp."]

vegetables = ["age", "gender", "fnlwgt", "education", "work-class", "marital-status", "occupation", "relationship",
              "race", "capital-gain", "capital-loss", "hours-per-week", "native-country"]
# vegetables = ["age", "gender", "fnlwgt", "education", "work", "marital", "occupation", "relationship",
#               "race", "capitalg", "capitall", "whours", "country"]

# vegetables = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

farmers = ["age", "gender", "fnlwgt", "education", "work-class", "marital-status", "occupation", "relationship",
           "race", "capital-gain", "capital-loss", "hours-per-week", "native-country"]
#
# farmers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]


# harvest = np.array([[0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
#                     [2.4, 0.0, 4.0, 1.0, 2.7, 0.0, 0.0],
#                     [1.1, 2.4, 0.8, 4.3, 1.9, 4.4, 0.0],
#                     [0.6, 0.0, 0.3, 0.0, 3.1, 0.0, 0.0],
#                     [0.7, 1.7, 0.6, 2.6, 2.2, 6.2, 0.0],
#                     [1.3, 1.2, 0.0, 0.0, 0.0, 3.2, 5.1],
#                     [0.1, 2.0, 0.0, 1.4, 0.0, 1.9, 6.3]])
#
# print harvest


co = np.array(np.around(coefs2, 2))
print "here = ", co
print co.shape

fig, ax = plt.subplots()
im = ax.imshow(co)

# We want to show all ticks...
ax.set_xticks(np.arange(len(farmers)))
ax.set_yticks(np.arange(len(vegetables)))
# ... and label them with the respective list entries
ax.set_xticklabels(farmers)
ax.set_yticklabels(vegetables)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(vegetables)):
    for j in range(len(farmers)):
        text = ax.text(j, i, co[i, j],
                       ha="center", va="center", color="w")

ax.set_title("Harvest of local farmers (in tons/year)")
fig.tight_layout()
# plt.tight_layout()
plt.show()
