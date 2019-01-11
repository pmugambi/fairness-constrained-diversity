from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import helpers
import prepare_adult_data as pad


# x: diversification attribute, y: sensitive attribute
def compute_correlation(x, y):
    return stats.pearsonr(x, y)


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


coefs = np.corrcoef([ages_list, genders_list, fnlwgts_list, educations_list, workclasses_list, maritalstatuses_list,
                   occupations_list, relationships_list, races_list, capital_gains_list, capital_losses_list,
                   hours_per_weeks_list, native_countries_list])

coefs2 = np.copy(coefs)

np.fill_diagonal(coefs2, np.nan)

print coefs.max()
print coefs.min()

min_distances = np.nanmin(coefs2, axis=1)
max_distances = np.nanmax(coefs2, axis=1)

print min(min_distances)
print max(max_distances)

# attributes_x = ["age", "gender", "fnlwgt", "education", "work-class", "marital-status", "occupation", "relationship",
#               "race", "capital-gain", "capital-loss", "hours-per-week", "native-country"]

attributes_x = ["age", "gender", "fnlwgt", "educ", "work", "marital", "occup", "r_ship", "race", "c_gain", "c_loss",
                "hours", "country"]

attributes_y = ["age", "gender", "fnlwgt", "education", "work-class", "marital-status", "occupation", "relationship",
           "race", "capital-gain", "capital-loss", "hours-per-week", "native-country"]


co = np.array(np.around(coefs2, 2))

fig, ax = plt.subplots()
im = ax.imshow(co)

# We want to show all ticks...
ax.set_xticks(np.arange(len(attributes_x)))
ax.set_yticks(np.arange(len(attributes_y)))
# ... and label them with the respective list entries
ax.set_xticklabels(attributes_x)
ax.set_yticklabels(attributes_y)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=90, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(attributes_x)):
    for j in range(len(attributes_y)):
        text = ax.text(j, i, co[i, j],
                       ha="center", va="center", color="w")

ax.set_title("Adult data attributes correlation")
fig.savefig("./data/results/variable_correlations")
# plt.tight_layout()
plt.show()
