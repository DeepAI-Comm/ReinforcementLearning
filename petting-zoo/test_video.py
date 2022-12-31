import os, sys
from pettingzoo.butterfly import pistonball_v6
import matplotlib.pyplot as plt
from IPython.display import clear_output

from video_func import KhoiRecordVideo

env = pistonball_v6.env(render_mode = 'rgb_array')
env.reset()
# plt.imshow(env.render())

epochs = 0

terminated = False

vid_rec = KhoiRecordVideo(env = env, video_folder = os.getcwd() + "/video")
while not terminated:
    print(epochs)
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        action = None if termination or truncation else env.action_space(agent).sample()  # this is where you would insert your policy
        vid_rec.step(action)
    
    terminated = termination

    epochs += 1
    
    if epochs == 5000:
        break
    
    clear_output(wait=True)

env.close()