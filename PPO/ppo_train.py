from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from dataclasses import dataclass
import torch
import torch.nn as nn
import random
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Optional, Union, Tuple


class PromptDataset(Dataset):
    def __init__(self, prompts, tokenizer, apply_chat_template=False):
        self.prompts = prompts
        self.tokenizer = tokenizer
        self.final_prompts = []

        for prompt in prompts:
            if apply_chat_template:
                content = [{"role": "user", "content": prompt}]
                prompt = self.tokenizer.apply_chat_template(content, tokenize=False, add_generation_prompt=True)
            else:
                prompt = self.tokenizer.bos_token + prompt
            
            self.final_prompts.append(prompt)
    
    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, index):
        return self.final_prompts[index]
        

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

class ExperienceBuffer:
    def __init__(self, limit):
        self.limit = limit
        self.buffer = []
    
    def append(self, experiences):
        batch = [{} for _ in range(len(experiences))]
        keys = (
            "seqs",
            "action_log_probs",
            "values",
            "returns",
            "advantages",
            "attention_mask",
            "action_mask",
            "num_actions"
        )

        for key in keys:
            for i, experience in enumerate(experiences):
                value = getattr(experience, key)
                batch[i][key] = value
        
        self.buffer.extend(batch)
        if len(self.buffer) >= self.limit:
            self.buffer = self.buffer[len(self.buffer)-self.limit:]
    
    def get_batches(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def clear(self):
        self.buffer = []
    
    def __len__(self):
        return len(self.buffer)
    
    def __getitem__(self, index):
        return self.buffer[index]

@dataclass
class Samples:
    seqs: torch.Tensor
    attention_mask: Optional[torch.LongTensor]


def train():
    # 初始化经验池
    buffer = ExperienceBuffer(limit=100)
    steps = 0
    for episode in range(episodes):
        for rand_prompts in prompts_dataloader:
            # 生成样本（获取模型推理结果）
            samples = 
            # 生成经验（获取优势、奖励、回报等）

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
    optimizer_actor = torch.optim.Adam(actor_model.parameters(), lr=0.00005)
    optimizer_critic = torch.optim.Adam(critic_model.parameters(), lr=0.00005)

    # 填充方式为左填充
    actor_tokenizer.padding_side = 'left'
    eos_token_id = actor_tokenizer.eos_token_id
    pad_token_id = actor_tokenizer.pad_token_id

    prompt_list = [
        '请问1+1等于多少？',
        'PowerShell，如何知道BIOS中的虚拟化是否已禁用',
        '为什么人们喜欢在水族馆里游泳，而不是在游泳池里？',
        '你是一位营销专家。为Instagram reels写30个带有营销技巧的脚本。',
        '你是一位营销专家。为Instagram reels写30个带有营销技巧的脚本。',
        '你是一位营销专家。为Instagram reels写30个带有营销技巧的脚本。',
        '为什么所有的镜子都是矩形的？',
        '我们在受感染的植物根部可以找到哪一种，臭氧还是金子？'
    ]

    prompts_dataset = PromptDataset(prompt_list, actor_tokenizer, apply_chat_template=True)
    prompts_dataloader = DataLoader(prompts_dataset, batch_size=rollout_batch_size, shuffle=True)

    train()