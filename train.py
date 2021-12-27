from Actor_Critic_for_JSP.JSP_env import JSP_Env,Gantt
import matplotlib.pyplot as plt
from Actor_Critic_for_JSP.Dataset.data_extract import change
from Actor_Critic_for_JSP.action_space import Dispatch_rule
from Actor_Critic_for_JSP.Agent.Agent import Agent

def main(Agent,env,batch_size):
    Reward_total = []
    C_total = []
    rewards_list = []
    C = []

    episodes = 8000
    print("Collecting Experience....")
    for i in range(episodes):
        print(i)
        state,done = env.reset()
        ep_reward = 0
        while True:

            action = Agent.choose_action(state)

            a=Dispatch_rule(action,env)
            try:
                next_state, reward, done = env.step(a)
            except:
                print(action,a)

            Agent.store_transition(state, action, reward, next_state)
            ep_reward += reward
            if Agent.memory_counter >= batch_size:
                Agent.learn()
                if done and i%1==0:
                    ret, f, C1, R1 = evaluate(i,Agent,env)
                    Reward_total.append(R1)
                    C_total.append(C1)
                    rewards_list.append( ep_reward)
                    C.append(env.C_max())
            if done:
                # Gantt(env.Machines)
                break
            state = next_state
    x = [_ for _ in range(len(C))]
    plt.plot(x, rewards_list)
    # plt.show()
    plt.plot(x, C)
    # plt.show()
    return Reward_total,C_total

def evaluate(i,Agent,env):
    returns = []
    C=[]
    for  total_step in range(10):
        state, done = env.reset()
        ep_reward = 0
        while True:
            action = Agent.choose_action(state)
            a = Dispatch_rule(action, env)
            try:
                next_state, reward, done = env.step(a)
            except:
                print(action,a)
            ep_reward += reward
            if done == True:
                fitness = env.C_max()
                C.append(fitness)
                break
        returns.append(ep_reward)
    print('time step:',i,'','Reward ：',sum(returns)/10 ,'','C_max:',sum(C) /10)
    return sum(returns) / 10,sum(C) /10,C,returns


if __name__ == '__main__':
    import pickle
    import os

    n, m, PT, MT = change('la', 16)

    f=r'.\result\la'
    if not os.path.exists(f):
        os.mkdir(f)
    f1=os.path.join(f,'la'+'16')
    if not os.path.exists(f1):
        os.mkdir(f1)
    print(n, m, PT, MT)
    env = JSP_Env(n, m, PT, MT)
    # (0,0)CNN+FNN+DQN (1,0):CNN+Dueling network+DQN (0,1):CNN+FNN+DDQN (1,1):CNN+Dueling network+DDQN
    agent=Agent(env.n,env.O_max_len,1,1)
    Reward_total,C_total=main(agent,env,100)
    print(os.path.join(f1, 'C_max' + ".pkl"))
    with open(os.path.join(f1, 'C_max' + ".pkl"), "wb") as f2:
        pickle.dump(C_total, f2, pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(f1, 'Reward' + ".pkl"), "wb") as f3:
        pickle.dump(Reward_total, f3, pickle.HIGHEST_PROTOCOL)