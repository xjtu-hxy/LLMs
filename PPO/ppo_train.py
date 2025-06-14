from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from dataclasses import dataclass
import torch
import torch.nn as nn
import random
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter


class Critic(nn.Module):
    #价值模型（评论家模型）
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.base_model.eval()
        self.value_head = nn.Linear(base_model.config.hidden_size, 1)
    
    def forward(self, input_ids, attention_mask, num_actions):
        hidden_state = self.base_model(input_ids, attention_mask=attention_mask).last_hidden_state
        value_model_output = self.value_head(hidden_state)
        values = value_model_output.squeeze(-1)[:, -num_actions:]
        return values


if __name__ == '__main__':
    device = ' cuda' if torch.cuda.is_available() else 'cpu'
    # 一共迭代多少轮
    episodes = 3
    # 生成一次经验，训练的轮数
    max_epochs = 5
    # 一次从提示词数据集中取多少条数据用于生成经验
    rollout_batch_size = 8
    # 一次取多少条数据生成经验（生成经验需要多个模型推理，对显存要求高）
    micro_rollout_batch_size = 2
    # 一个提示词生成多少个样本
    n_samples_per_prompt = 2
    # 生成的最大长度，相当于最大动作数，数值越大，模型探索的可能性越多
    max_new_tokens = 50
    # 最大长度
    max_length = 256
    # 实际训练的batch_size大小，一次取多少条数据用于更新参数
    micro_train_batch_size = 2
    #记录日志
    writer = SummaryWriter('./runs')
    # 模型
    model_dir = '/data/ppo/LLMs/PPO/Qwen/Qwen2___5-0___5B-Instruct'
    # 策略模型
    actor_model = AutoModelForCausalLM.from_pretrained(model_dir).to(device)
    # 参考模型
    ref_model = AutoModelForCausalLM.from_pretrained(model_dir).to(device)

    reward_model_dir = ''
    # 奖励模型
    reward_model = AutoModelForSequenceClassification.from_pretrained(reward_model_dir).to(device)
    actor_tokenizer = AutoTokenizer.from_pretrained(model_dir)
    reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_dir)
    # 价值模型
    critic_model = Critic(actor_model.base_model).to(device)

    # 初始化优化器
    optimizer_act