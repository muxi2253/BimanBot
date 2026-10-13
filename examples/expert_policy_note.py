"""
Gym 环境添加专家策略接口
"""

def get_expert_action(self):
    """
    获取规则策略的专家动作

    注意：这需要在 gym_env.py 中实现
    目前返回随机动作作为占位符

    TODO: 集成 public_demo.py 的规则策略
    """
    # 临时实现：返回随机动作
    return self.action_space.sample()
