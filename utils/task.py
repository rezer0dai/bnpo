class Task:
    def __init__(self, RLTask, rew_min, do_assess=True, goals=None):
        self.ENV = RLTask()
        self.rew_min = rew_min

        self.goals = goals
        self.do_assess = do_assess

    def reset(self, agent, seed, learn_mode):
        einfo = self.ENV.reset(agent, seed, learn_mode)
        if self.do_assess:
            self.goals = einfo.goals # this can be problematic in HRL settings
            #  self.goals = self.ENV.goals
        return einfo.states

    def goal(self):
        return self.goals

    def step(self, e_pi, t_pi, learn_mode):
        actions = e_pi.action()
        einfo = self.ENV.step(e_pi.params(False))

        if self.do_assess:
            self.goals = einfo.goals
#        goods = [True] * len(einfo.rewards)
        goods = einfo.goods

        # temporararely - as we recalc now all rews when sampling
        rewards = einfo.rewards if not learn_mode else einfo.custom_rewards

        # actions reflect finally choosend actions!!
        pi = e_pi.q_action(einfo.actions)
        log_prob = t_pi.log_prob(einfo.actions)
        return log_prob, pi, einfo.actions, einfo.states, rewards, einfo.dones, goods

    def goal_met(self, rewards):
        return rewards > self.rew_min
