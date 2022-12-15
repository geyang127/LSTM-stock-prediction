
import pandas as df
import matplotlib.pyplot as plt
def agent_trade(model,x_test,test_ds,y_test,window_size):
    model.evaluate(test_ds)
    y_pred = model.predict(x_test)
    df_time = df.index[-len(y_test):]
    is_hold=0
    reward = 0.0
    return_list = []
    reward_list=[]
    for i in range(len(window_size,y_pred)):
        return_list.append(y_test[i]-y_test[i-window_size])
        if  (y_pred[i]-y_pred[i-window_size])/y_pred[i-window_size]<=0.02:
            if is_hold ==1:
                reward+=(y_test[i]-y_test[i-window_size])/y_test[i-window_size]
                reward_list.append(reward)
                is_hold =0
        else:
            is_hold=1
    return return_list,reward_list

df_time = df.index[-len(y_test):]
fig = plt.figure(figsize=(10, 5))
axes = fig.add_subplot(111)
axes.plot(df_time, gru_reward, 'b-', label='stacked GRU')
axes.plot(df_time, reward_baseline 'y-', label='hold and sell')
axes.plot(df_time, lstm_reward), 'r--', label='stacked LSTM')
axes.set_xticks(df_time[::50])
plt.legend()
plt.grid()
plt.show()
