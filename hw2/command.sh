# Experiment 1 (CartPole-v0)
uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 1000 \
    --exp_name cartpole
uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 1000 \
    -rtg --exp_name cartpole_rtg
uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 1000 \
    -na --exp_name cartpole_na
uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 1000 \
    -rtg -na --exp_name cartpole_rtg_na
uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 4000 \
    --exp_name cartpole_lb
uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 4000 \
    -rtg --exp_name cartpole_lb_rtg
uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 4000 \
    -na --exp_name cartpole_lb_na
uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 4000 \
    -rtg -na --exp_name cartpole_lb_rtg_na

# Experiment 2 (HalfCheetah-v4)
# No baseline
uv run src/scripts/run.py --env_name HalfCheetah-v4 -n 100 -b 5000 -eb 3000 -rtg \
    --discount 0.95 -lr 0.01 --exp_name cheetah
# Baseline
uv run src/scripts/run.py --env_name HalfCheetah-v4 -n 100 -b 5000 -eb 3000 -rtg \
    --discount 0.95 -lr 0.01 --use_baseline -blr 0.01 -bgs 5 --exp_name cheetah_baseline

# effects of baseline gradient steps
uv run src/scripts/run.py --env_name HalfCheetah-v4 -n 100 -b 5000 -eb 3000 -rtg \
    --discount 0.95 -lr 0.01 --use_baseline -blr 0.01 -bgs 10 --exp_name cheetah_baseline_bgs10
uv run src/scripts/run.py --env_name HalfCheetah-v4 -n 100 -b 5000 -eb 3000 -rtg \
    --discount 0.95 -lr 0.01 --use_baseline -blr 0.01 -bgs 2 --exp_name cheetah_baseline_bgs2
uv run src/scripts/run.py --env_name HalfCheetah-v4 -n 100 -b 5000 -eb 3000 -rtg \
    --discount 0.95 -lr 0.01 --use_baseline -blr 0.01 -bgs 15 --exp_name cheetah_baseline_bgs15

# effects of baseline learning rate
uv run src/scripts/run.py --env_name HalfCheetah-v4 -n 100 -b 5000 -eb 3000 -rtg \
    --discount 0.95 -lr 0.01 --use_baseline -blr 0.001 -bgs 5 --exp_name cheetah_baseline_blr0.001
uv run src/scripts/run.py --env_name HalfCheetah-v4 -n 100 -b 5000 -eb 3000 -rtg \
    --discount 0.95 -lr 0.01 --use_baseline -blr 0.1 -bgs 5 --exp_name cheetah_baseline_blr0.1

# Experiment 3 (LunarLander-v2)
# lambda = {0, 0.95, 0.98, 0.99, 1}
uv run src/scripts/run.py --env_name LunarLander-v2 --ep_len 1000 --discount 0.99 \
    -n 200 -b 2000 -eb 2000 -l 3 -s 128 -lr 0.001 --use_reward_to_go --use_baseline \
    --gae_lambda 1 --exp_name lunar_lander_lambda1
uv run src/scripts/run.py --env_name LunarLander-v2 --ep_len 1000 --discount 0.99 \
    -n 200 -b 2000 -eb 2000 -l 3 -s 128 -lr 0.001 --use_reward_to_go --use_baseline \
    --gae_lambda 0.99 --exp_name lunar_lander_lambda0.99
uv run src/scripts/run.py --env_name LunarLander-v2 --ep_len 1000 --discount 0.99 \
    -n 200 -b 2000 -eb 2000 -l 3 -s 128 -lr 0.001 --use_reward_to_go --use_baseline \
    --gae_lambda 0.98 --exp_name lunar_lander_lambda0.98
uv run src/scripts/run.py --env_name LunarLander-v2 --ep_len 1000 --discount 0.99 \
    -n 200 -b 2000 -eb 2000 -l 3 -s 128 -lr 0.001 --use_reward_to_go --use_baseline \
    --gae_lambda 0.95 --exp_name lunar_lander_lambda0.95
uv run src/scripts/run.py --env_name LunarLander-v2 --ep_len 1000 --discount 0.99 \
    -n 200 -b 2000 -eb 2000 -l 3 -s 128 -lr 0.001 --use_reward_to_go --use_baseline \
    --gae_lambda 0 --exp_name lunar_lander_lambda0

# Experiment 4 (InvertedPendulum). 
uv run src/scripts/run.py --env_name InvertedPendulum-v4 -n 100 -b 5000 -eb 1000 \
    --exp_name pendulum    

# sota fornow
uv run src/scripts/run.py --env_name InvertedPendulum-v4 -n 100 -b 5000 -eb 1000 \
    -l 4 -s 256 -rtg --discount 0.95 -lr 0.001 --use_baseline -blr 0.01 -bgs 25 \
    -na --exp_name pendulum_na_l4_s256_rtg_discount0.95_lr0.001_blr0.01_bgs25

uv run src/scripts/run.py --env_name InvertedPendulum-v4 -n 100 -b 5000 -eb 1000 \
    -l 4 -s 256 -rtg --discount 0.95 -lr 0.001 --use_baseline -blr 0.01 -bgs 20 \
    -na --exp_name pendulum_na_l4_s256_rtg_discount0.95_lr0.001_blr0.01_bgs20

uv run src/scripts/run.py --env_name InvertedPendulum-v4 -n 100 -b 5000 -eb 1000 \
    -l 4 -s 256 -rtg --discount 0.98 -lr 0.001 --use_baseline -blr 0.01 -bgs 35 \
    -na --exp_name pendulum_na_l4_s256_rtg_discount0.98_lr0.001_blr0.01_bgs35

uv run src/scripts/run.py --env_name InvertedPendulum-v4 -n 100 -b 5000 -eb 1000 \
    -l 4 -s 256 -rtg --discount 0.95 -lr 0.001 --use_baseline -blr 0.1 -bgs 25 \
    -na --exp_name pendulum_na_l4_s256_rtg_discount0.95_lr0.001_blr0.1_bgs25

uv run src/scripts/run.py --env_name InvertedPendulum-v4 -n 100 -b 5000 -eb 1000 \
    -l 4 -s 256 -rtg --discount 0.99 -lr 0.001 --use_baseline -blr 0.01 -bgs 15 \
    -na --exp_name pendulum_na_l4_s256_rtg_discount0.99_lr0.001_blr0.01_bgs15


