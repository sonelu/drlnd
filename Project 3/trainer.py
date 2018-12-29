import time
import json
import numpy as np
from collections import deque


def trainer(env, brain_name,
            agent, 
            n_episodes=1000, t_max=10800,
            print_every=None,
            save=True,
            eps_start=1.0, eps_min=0.05, eps_decay=0.99,
            threshold=0.5):
    """A help function that will handle the interaction with the environment
    and manage the agent.
    
    Params:
    ======
    env (Unity environment): the environment to work with
    brain_name (string): the name of the brain in the environment
    agent (and Agent class): the agent that will be trained; the Agent
            must implement the methods `act`, `step` and `save`
    n_episodes (int): maximum number of episodes the trainer will run
    t_max (int): maximum duration to run the training is seconds
    save (Bool): saves the model with the best score
    eps_start, eps_min (float): parameters for noise
    eps_decay (float): dacay factor for applying noise
    threshold (float): the treshold for considering the problem solved
    """
    # print string for convenience
    print_str = '\rEpisode: {:4d} Score: {:.3f} Average: {:.3f} Steps: {:4d} Duration: {:.1f}s Running: {:.1f}s'

    # variable to store the results
    last_avg = 0
    scores_window = deque(maxlen=100)
    tot_dur = 0.0
    eps = eps_start
    
    # for saving things
    # run identification
    runID = time.strftime('%Y%m%d%H%M%S', time.localtime())
    print('runID:', runID)
    results_csv = 'results/' + runID + '.csv'
    results_info = 'results/' + runID + '.info'
    model_pth = 'models/' + runID + '.pth'
    model_info = 'models/' + runID + '.info'
    
    # write the info
    exclude = ['net', 'critic_optimizer', 'actor_optimizer', 
               'memory', 'noise', 'device']
    info = {k:v for (k,v) in agent.__dict__.items() if k not in exclude}
    with open(results_info, 'w') as f:
        json.dump(info, f)
    # write info for model
    with open(model_info, 'w') as f:
        json.dump(info, f)
    
    for i_episode in range(1, n_episodes+1):
        steps = 0
        start = time.time()
        env_info = env.reset(train_mode=True)[brain_name]
        obs = env_info.vector_observations
        scores = np.zeros(len(env_info.agents))

        # run one episode
        while True:
            
            # determine action
            actions = agent.act(obs, eps)
            steps += 1
            
            # take the action; get the environment response
            env_info = env.step(actions)[brain_name]
            next_obs = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            
            # ask agent to process the results
            agent.step(obs, actions, rewards, next_obs, dones)

            # cumulate and move to next
            obs = next_obs
            scores += rewards
            if np.any(dones):
                # unless episode complete
                break
                
        # at the end of the episode keep track of time and scores
        duration = time.time() - start
        tot_dur += duration
        scores_window.append(np.max(scores))
        avg = np.mean(scores_window)

        # print current numbers
        with open(results_csv,'a') as f:
            f.write('{:d},{:.5f},{:5f},{:5f}\n'.format(i_episode, 
                                np.max(scores), avg, tot_dur))

        if print_every is not None and i_episode % print_every == 0:
            end = '\n'
        else:
            end = ''
        print(print_str.format(i_episode, np.max(scores), avg, 
                steps, duration, tot_dur), end=end)

        if save and avg > last_avg:
            agent.save(model_pth)
            last_avg = avg

        # if solved
        if avg >= threshold:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}\t In In {:.1f}s'\
                  .format(i_episode, avg, tot_dur))
            break
            
        if tot_dur > t_max:
            break
            
        eps = eps * eps_decay
        eps = max(eps, eps_min)

    # check if we finished all episodes but not found a solution
    if avg < threshold:
        print('\nFailed to converge in {:d} episodes.\tAverage Score: {:.2f}\t In {:.1f}'\
              .format(n_episodes, avg, tot_dur))

    # return results for analysis
    return runID