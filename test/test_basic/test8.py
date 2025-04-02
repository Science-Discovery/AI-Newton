#%%
from aiphy import stupid_analysis, Knowledge

knowledge = Knowledge.default()
#%%
print("Round 0")
exp = stupid_analysis(knowledge, "motion0")
knowledge.print_concepts()
print("Round 1")
exp = stupid_analysis(knowledge, "motion0")
knowledge.print_concepts()
print("Round 2")
exp = stupid_analysis(knowledge, "motion0")
knowledge.print_concepts()
#%%
print("Round 3")
exp = stupid_analysis(knowledge, "motion")
knowledge.print_concepts()
#%%
print("Round 4")
exp = stupid_analysis(knowledge, "oscillation")
knowledge.print_concepts()
print("Round 5")
exp = stupid_analysis(knowledge, "oscillation")
knowledge.print_concepts()
#%% print conclusions
knowledge.print_conclusions()