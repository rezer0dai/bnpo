import torch
import torch.optim as optim
import utils.policy as policy

from utils.ngd_opt import NGDOptim, callback_with_logits

class BrainOptimizer:
    def __init__(self, brain, desc):
        self.bellman = desc.bellman
        self.natural = desc.natural
        self.brain = brain

        if self.natural:
            self.actor_optimizer = NGDOptim(
                brain.ac_explorer.actor_parameters(), lr=desc.lr_actor, momentum=.7, nesterov=True)
        else:
            self.actor_optimizer = optim.Adam(
                brain.ac_explorer.actor_parameters(), lr=desc.lr_actor)

        self.clip = desc.ppo_eps is not None
        if not self.clip:
            self.loss = policy.DDPGLoss(advantages=True, boost=False)
        else:
            self.loss = policy.PPOLoss(eps=desc.ppo_eps, advantages=True, boost=False)

    def __call__(self, qa, td_targets, probs, actions, dist, _eval):
        assert self.bellman or self.clip, "ppo can be only active when clipped ( in our implementation )!"

        if not self.clip:#vanilla ddpg
            pi_loss = self.loss(qa, td_targets, None, None)
        elif self.bellman:#DDPG with clip
            pi_loss = self.loss(td_targets, qa,
                dist.log_prob(actions).mean(1).detach(), probs.mean(1))
        else:#PPO
            pi_loss = self.loss(qa.detach(), td_targets.detach(),
                probs.mean(1), dist.log_prob(actions).mean(1))

        # descent
        pi_loss = -pi_loss.mean()

        # learn!
        self.brain.backprop(
                self.actor_optimizer,
                pi_loss,
                self.brain.ac_explorer.actor_parameters(),
                # next is for natural gradient experimenting!
                None if not self.natural else callback_with_logits(self.actor_optimizer, dist, _eval),
                just_grads=False)

        return pi_loss
