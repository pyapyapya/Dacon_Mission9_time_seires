
# coding: utf-8

# In[1]:


import numpy as np
import seaborn as sns
import pandas as pd
import datetime
import sys

from glob import glob
from tqdm import tqdm
import pmdarima as pm

from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima_model import ARIMA
from pmdarima.arima import auto_arima
pd.options.display.max_rows = 50
pd.options.display.max_columns = 200
desired_width = 320

pd.set_option('display.width',desired_width)
pd.set_option('display.max_rows',500)
pd.set_option('display.max_columns',500)
pd.set_option('display.width', 1300)
pd.set_option('display.expand_frame_repr', False)

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# train_df = pd.read_csv('F:\\funda_train.csv')
# test_df = pd.read_csv('F:\\submission.csv')
train_df = pd.read_csv('E:\\funda_train.csv')
test_df = pd.read_csv('E:\\submission.csv')
combine = [train_df, test_df]
dump_df = train_df

print(train_df.shape)
print(train_df.head(10))
print(train_df.tail(10))


# In[3]:


train_df.info()


# In[4]:


test_df


# In[5]:


train_df['region'].unique()


# In[6]:


train_df['type_of_business'].unique()


# In[7]:


train_df.groupby('type_of_business')['amount'].mean()


# In[8]:


train_df.groupby('region')['amount'].mean()


# In[9]:


train_df.groupby('store_id')['amount'].mean()


# In[10]:


def pltplot(x_data, y_data, title, ylabel):
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    
    plt.plot(x_data, y_data)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.show()


# In[11]:


def mk_xdata(len_xdata):
    return list(range(len_xdata))


# In[12]:


len_storeid = len(train_df['store_id'].unique())
# x_data = mk_xdata(len_storeid)
# x_data = np.array(x_data)
x_data = np.arange(len_storeid)  # <-- 여-기
y_data = np.array(train_df.groupby('store_id')['amount'].mean())
pltplot(x_data, y_data, 'store_id', 'amount')


# In[13]:


df_business = train_df[['type_of_business', 'amount']]
df_business = df_business.dropna(axis=0, subset=['type_of_business'])
lst_business  = df_business['type_of_business'].unique().tolist()

len_business = len(lst_business)
total_len = len(train_df)
business_sum = [0] * len_business

for total_idx in tqdm(range(total_len)):
    try:
        find_business = df_business['type_of_business'][total_idx]
    except KeyError:
        continue
    
    for business in lst_business:
        if business == find_business:
            idx = lst_business.index(business)
            business_sum[idx] += df_business['amount'][total_idx]


# In[14]:


# for i in range(len_business):
#     a = business_sum[i]


# In[15]:


df_region = train_df[['region', 'amount']]
df_region = df_region.dropna(axis=0, subset=['region'])
lst_region = df_region['region'].unique().tolist()
len_region = len(lst_region)
region_sum = [0] * len_region
for total_idx in tqdm(range(total_len)):
    try:
        find_region = df_region['region'][total_idx]
    except KeyError:
        continue
    for region in lst_region:
        if region == find_region:
            idx = lst_region.index(region)
            region_sum[idx] += df_region['amount'][total_idx]


# In[16]:


for i in range(len_region):
    print(int(region_sum[i]))


# In[17]:


x_data = np.arange(len_business)
y_data = np.array(business_sum)
pltplot(x_data, y_data, 'type_of_business', 'amount')


# In[18]:


# 상위 4개의 feature를 뽑아내는 과정

business_top = []
tmp_lst = []
tmp_lst = sorted(business_sum, reverse = True)
for i in range(4):
    idx = business_sum.index(tmp_lst[i])
    business_top.append(idx)
business_top


# In[19]:


x_data = np.arange(len_region)
y_data = np.array(region_sum)
pltplot(x_data, y_data, 'region', 'amount')


# In[20]:


# 상위 17개의 feature를 봅아내는 과정

region_top = []
tmp_lst = []
tmp_lst = sorted(region_sum, reverse = True)
for i in range(17):
    idx = region_sum.index(tmp_lst[i])
    region_top.append(idx)
region_top


# In[21]:


train_df.transacted_date = train_df.transacted_date.str[:-3]


# In[22]:


lst_store_id = train_df['store_id'].unique()


# In[23]:


store_month_amount = pd.pivot_table(train_df, index=["store_id", "transacted_date"], values=["amount"], aggfunc=[np.mean, np.sum, len])
store_month_amount


# In[24]:


# store_month_amount.xs((0, '2016-06'))
type(store_month_amount.xs(0)['sum'].as_matrix())


# In[25]:


n_stores = len_storeid
train = [store_month_amount.xs(i)['sum'].as_matrix() for i in tqdm(lst_store_id)] 
lst_month_amount = np.array([store_month_amount.xs(i)['sum'].as_matrix() for i in tqdm(lst_store_id)])
# 개 졸 려
# date 값을 채우거나 버리기


# In[26]:


del_lst = []
total_month = len(lst_month_amount[0])
for i in tqdm(range(len(lst_month_amount))):
    if len(lst_month_amount[i]) != total_month:
        del_lst.append(i)


# In[27]:


len(del_lst)


# In[28]:


tmp = []
for i in range(len(lst_month_amount)):
    tmp.append(len(lst_month_amount[i]))
tmp


# In[29]:


pd.pivot_table(train_df, index=["transacted_date"], values=["amount"], aggfunc=[np.mean, np.sum, len])


# In[30]:


pd.pivot_table(train_df, index=["transacted_date", "store_id"], values=["amount"], aggfunc=[np.max, np.min, len])


# In[31]:


# len_card = len(dump_df['card_company'].unique())
# cnt_card = [0] * (len_card)
# for card in tqdm(dump_df['card_company']):
#     idx = ord(card) - 97
#     cnt_card[idx] = cnt_card[idx] + 1
# cnt_card


# In[32]:


i = 0
lst_card_company = []
for card in tqdm(dump_df['card_company']):
    # dump_df['card_company'][i] = ord(card) - 96
    #lst_card_company.append([])
    lst_card_company.append(ord(card) - 96)
    # i = i + 1


# In[33]:


dump_df['card_company'] = lst_card_company


# In[34]:


scaler = MinMaxScaler()
dump_df['card_company'] = scaler.fit_transform(np.reshape(lst_card_company, (-1, 1)))


# In[35]:


dump_df['type_of_business'] = dump_df['type_of_business'].fillna(0)
business_to_num = []
lst_business  = dump_df['type_of_business'].unique().tolist()
for business in tqdm(dump_df['type_of_business']):
    for type_business in lst_business:
        if business == type_business:
            idx = lst_business.index(business)
            business_to_num.append(idx)
business_to_num


# In[36]:


dump_df['type_of_business'] = scaler.fit_transform(np.reshape(business_to_num, (-1, 1)))
dump_df


# In[37]:


dump_df['region'] = dump_df['region'].fillna(0)
region_to_num = []
lst_to_region  = dump_df['region'].unique().tolist()
for region in tqdm(dump_df['region']):
    for compare_region in lst_to_region:
        if region == compare_region:
            idx = lst_to_region.index(region)
            region_to_num.append(idx)


# In[38]:


dump_df['region'] = scaler.fit_transform(np.reshape(region_to_num, (-1, 1)))
dump_df


# In[39]:


# lst_amount = []
# for amount in tqdm(train_df['amount']):
#     amount = 
#     lst_amount.append(amount)
# dump_df['amount'] = scaler.fit_transform(np.reshape(lst_amount, (-1, 1)))
# lst_amount


# In[40]:


# m_min = abs(min(lst_amount))
# m_max = max(lst_amount)
# cnt = [0] * (m_min + m_max + 1)
# for i in lst_amount:
#     cnt[m_min + i] = cnt[m_min + i] + 1
# cnt


# In[41]:


# x_data = np.arange(m_min+m_max+1)
# y_data = np.array(cnt)
# pltplot(x_data, y_data, 'distribution', 'amount')


# In[42]:


len(train_df['transacted_time'])


# In[43]:


tmp = [int(train_df['transacted_time'][i].split(':')[0]) // 6 for i in tqdm(range(len(train_df['transacted_time'])))]

dump_df['transacted_time'] = tmp


# In[44]:


dump_df = dump_df.drop(['card_id', 'installment_term'], axis=1)
dump_df


# In[45]:


model = ARIMA(lst_month_amount[0], order=(0, 1, 1))  # dum_df 가 pandas object에서 안되는겨
model_fit = model.fit(disp=-1)  
# model_fit = model.fit(trend='c',full_output=True, disp=1)
print(model_fit.summary())
# plt.plot(ts_log_diff)
# plt.plot(results_MA.fittedvalues, color='red')


# In[51]:


# # Plot residual errors
# month = pd.DataFrame(model_fit.resid)
# fig, ax = plt.subplots(1,2)
# residuals.plot(title="month", ax=ax[0])
# residuals.plot(kind='kde', title='Density', ax=ax[1])
# plt.show()


# In[47]:


arr = []
for element in train:
    for e in element:
        arr.append(e)
arr


# In[48]:


#building the model
model1 = auto_arima(arr, trace=True, error_action='ignore', suppress_warnings=True)


# In[55]:


model_fit1 = model1.fit(disp=-1)
model_fit1.plot_predict(dynamic=False)
plt.show()


# In[53]:


model_fit.plot_predict(dynamic=False)
plt.show()


# In[52]:


model1.fit(arr)

forecast = model1.predict(n_periods=36)
forecast = pd.DataFrame(forecast,index = 36,columns=['Prediction'])

#plot the predictions for validation set
plt.plot(arr, label='Train')
plt.plot(36, label='Valid')
plt.plot(forecast, label='Prediction')
plt.show()


# In[50]:


model = pm.auto_arima(arr, start_p=1, start_q=1,
                      test='adf',       # use adftest to find optimal 'd'
                      max_p=3, max_q=3, # maximum p and q
                      m=1,              # frequency of series
                      d=None,           # let model determine 'd'
                      seasonal=False,   # No Seasonality
                      start_P=0, 
                      D=0, 
                      trace=True,
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)

print(model.summary())


# In[ ]:


model = ARIMA(lst_month_amount[0], order=(1, 1, 1))
results_MA = model.fit(disp=-1)  
print(results_MA.summary())


# In[ ]:


model = ARIMA(lst_month_amount[0], order=(1, 1, 2))
results_MA = model.fit(disp=-1)  
print(results_MA.summary())


# In[ ]:


# features
# feat 1 : card_company a~z => 1 ~ 26 / max normalize -> --------8 comany done
# feat 2 : transacted_date year-month => 2016-06 ~ 2019-02 amount 0 => ----------- deleted
# feat 3 : transacted_time 2~10 / 10 ~ 18 / 18 ~ 02 => 0, 1, 2 encoding / max normalize 
# feat 4 : amount, amount    / 1e6 normalize
# feat 5 : region => 0 ~ 175 / max 값 normalize -------done
# feat 6 : type_of_business => above -------done
# 1 data : vector (1000 , 6) dims (seq_len, feat_dims)
# (2175, 1000, 6) data # (number_of_data, seq_len, feat_dims)

# model
# 1. deep learning
# 2. xgboost / etc

# https://www.kaggle.com/kashnitsky/topic-9-part-1-time-series-analysis-in-python
# https://www.kaggle.com/headsortails/wiki-traffic-forecast-exploration-wtf-eda

