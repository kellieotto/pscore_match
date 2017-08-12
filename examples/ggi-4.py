treated_turnout = imai.VOTED98[treatment == 1].mean()
control_turnout = imai.VOTED98[treatment==0].mean()
matched_control_turnout = data_matched.voted[data_matched.treatment==0].mean()
ATT = treated_turnout - control_turnout
matched_ATT = treated_turnout - matched_control_turnout
print(str("ATT: " + str(ATT)))
print(str("ATT after matching: " + str(matched_ATT)))