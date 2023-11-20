import matplotlib.pyplot as plt
import numpy as np 

gbf = np.load("different_trials/gbf_cmu.npy")
all_my_fuz = []
for i in range(11):   
    myfuz = np.load(f"different_trials/middle/myfuz_cmu{i}.npy") if i>0 else np.load(f"different_trials/middle/myfuz_cmu.npy")
    all_my_fuz.append(myfuz)
all_my_fuz = np.vstack(all_my_fuz)
#print(all_my_fuz.shape)
# all_sum = np.sum(all_my_fuz,axis=0).reshape(9999)
# print(all_sum.shape)
# all_mean = all_sum/11
all_mean = np.mean(all_my_fuz,axis = 0)
all_std = np.std(all_my_fuz,axis = 0)
first_std = all_mean + all_std
first_std2= all_mean - all_std
plt.plot(gbf)
plt.plot(all_mean)
plt.plot(first_std)
plt.plot(first_std2)
plt.title('Coverage over time')
plt.xlabel('# of inputs')
plt.ylabel('lines covered')
plt.legend(['Boosted Greybox',"My Fuzzer Mean Alpha = .1","My Fuzzer Mean + Std. Alpha = .1","My Fuzzer Mean - Std. Alpha = .1"]) 
plt.show()

all_my_fuz = []
for i in range(11):   
    myfuz = np.load(f"different_trials/too_small/myfuz_cmu{i}.npy") if i>0 else np.load(f"different_trials/too_small/myfuz_cmu.npy")
    all_my_fuz.append(myfuz)
all_my_fuz = np.vstack(all_my_fuz)
#print(all_my_fuz.shape)
# all_sum = np.sum(all_my_fuz,axis=0).reshape(9999)
# print(all_sum.shape)
# all_mean = all_sum/11
all_mean = np.mean(all_my_fuz,axis = 0)
all_std = np.std(all_my_fuz,axis = 0)
first_std = all_mean + all_std
first_std2= all_mean - all_std
plt.plot(gbf)
plt.plot(all_mean)
plt.plot(first_std)
plt.plot(first_std2)
plt.title('Coverage over time')
plt.xlabel('# of inputs')
plt.ylabel('lines covered')
plt.legend(['Boosted Greybox',"My Fuzzer Mean Alpha = .001","My Fuzzer Mean + Std. Alpha = .001","My Fuzzer Mean - Std. Alpha = .001"]) 
plt.show()

all_my_fuz = []
for i in range(11):   
    myfuz = np.load(f"different_trials/too_big/myfuz_cmu{i}.npy") if i>0 else np.load(f"different_trials/too_big/myfuz_cmu.npy")
    all_my_fuz.append(myfuz)
all_my_fuz = np.vstack(all_my_fuz)
#print(all_my_fuz.shape)
# all_sum = np.sum(all_my_fuz,axis=0).reshape(9999)
# print(all_sum.shape)
# all_mean = all_sum/11
all_mean = np.mean(all_my_fuz,axis = 0)
all_std = np.std(all_my_fuz,axis = 0)
first_std = all_mean + all_std
first_std2= all_mean - all_std
plt.plot(gbf)
plt.plot(all_mean)
plt.plot(first_std)
plt.plot(first_std2)
plt.title('Coverage over time')
plt.xlabel('# of inputs')
plt.ylabel('lines covered')
plt.legend(['Boosted Greybox',"My Fuzzer Mean Alpha = .6","My Fuzzer Mean + Std. Alpha = .6","My Fuzzer Mean - Std. Alpha = .6"]) 
plt.show()