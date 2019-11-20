def get_ave(train):
    #Steps Mean Constant
    mean_lds_steps = train[(train.cal_burn < 2500) & (train.cal_burn > 2000)].steps.mean()

    #Distance Mean Constant
    mean_lds_distance = train[(train.cal_burn < 2500) & (train.cal_burn > 2000)].distance.mean()

    #Floors Mean Constant
    mean_lds_floor = train[(train.cal_burn < 2500) & (train.cal_burn > 2000)].floors.mean()

    #Lightly Active Minutes Mean Constant
    mean_lds_light = train[(train.cal_burn < 2500) & (train.cal_burn > 2000)].min_active_light.mean()

    #Fairly Active Minutes Mean Constant
    mean_lds_fairly = train[(train.cal_burn < 2500) & (train.cal_burn > 2000)].min_active_fairly.mean()

    #Very Active Minutes Mean Constant
    mean_lds_very = train[(train.cal_burn < 2500) & (train.cal_burn > 2000)].min_active_very.mean()

    return mean_lds_steps, mean_lds_distance, mean_lds_floor, mean_lds_light, mean_lds_fairly, mean_lds_very