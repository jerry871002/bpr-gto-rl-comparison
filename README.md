# Comparing Exploitation-Based and Game Theory Optimal Based Approaches in a Multi-Agent Environment

In this project, we compared two algorithms
- **BPR**: exploitative style, a way of playing to identify and exploit imbalances in the strategies of your opponents.
- **MADDPG/M3DDPG**: game theory optimal (GTO) style, a way of playing a game that makes you 
unexploitable to your opponents.

Check [the report](https://drive.google.com/file/d/1AAI8e53vWmCa3qv-24zm3hL61XTQWDmy/view?usp=sharing) for more detail.

## Remarks

- `env.py` is the environment we developed to test the algorithms. You can interact with the environment by running `play_with_model.py`.
- `train/` folder contains the code we used to train our agent.
  - Notice that you may need to add `sys.path.append` to make `import env` works
- For the MADDPG/M3DDPG agents, we stored them as `pickle` objects after training for reuse.
