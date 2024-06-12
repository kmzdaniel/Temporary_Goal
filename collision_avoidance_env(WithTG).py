'''
Collision Avoidance Environment
Author: Michael Everett
MIT Aerospace Controls Lab
'''

import gym
import gym.spaces
import numpy as np
import itertools
import copy
import os
import inspect
import sys

from gym_collision_avoidance.envs import Config
from gym_collision_avoidance.envs.util import find_nearest, rgba2rgb, l2norm, makedirs
from gym_collision_avoidance.envs.visualize import plot_episode, animate_episode
from gym_collision_avoidance.envs.agent import Agent
from gym_collision_avoidance.envs.Map import Map
from gym_collision_avoidance.envs import test_cases as tc

from sympy import * # i add this

class CollisionAvoidanceEnv(gym.Env):
    """ Gym Environment for multiagent collision avoidance

    The environment contains a list of agents.

    :param agents: (list) A list of :class:`~gym_collision_avoidance.envs.agent.Agent` objects that represent the dynamic objects in the scene.
    :param num_agents: (int) The maximum number of agents in the environment.
    """

    # Attributes:
    #     agents: A list of :class:`~gym_collision_avoidance.envs.agent.Agent` objects that represent the dynamic objects in the scene.
    #     num_agents: The maximum number of agents in the environment.

    metadata = {
        # UNUSED !!
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):

        self.id = 0

        # Initialize Rewards
        self._initialize_rewards()

        # Simulation Parameters
        self.num_agents = Config.MAX_NUM_AGENTS_IN_ENVIRONMENT
        self.dt_nominal = Config.DT

        # Collision Parameters
        self.collision_dist = Config.COLLISION_DIST
        self.getting_close_range = Config.GETTING_CLOSE_RANGE

        # Plotting Parameters
        self.evaluate = Config.EVALUATE_MODE

        self.plot_episodes = Config.SHOW_EPISODE_PLOTS or Config.SAVE_EPISODE_PLOTS
        self.plt_limits = Config.PLT_LIMITS
        self.plt_fig_size = Config.PLT_FIG_SIZE
        self.test_case_index = 0

        self.set_testcase(Config.TEST_CASE_FN, Config.TEST_CASE_ARGS)

        self.animation_period_steps = Config.ANIMATION_PERIOD_STEPS

        # if Config.TRAIN_ON_MULTIPLE_AGENTS:
        #     self.low_state = np.zeros((Config.FULL_LABELED_STATE_LENGTH))
        #     self.high_state = np.zeros((Config.FULL_LABELED_STATE_LENGTH))
        # else:
        #     self.low_state = np.zeros((Config.FULL_STATE_LENGTH))
        #     self.high_state = np.zeros((Config.FULL_STATE_LENGTH))

        # Upper/Lower bounds on Actions
        self.max_heading_change = np.pi/3
        self.min_heading_change = -self.max_heading_change
        self.min_speed = 0.0
        self.max_speed = 1.0

        ### The gym.spaces library doesn't support Python2.7 (syntax of Super().__init__())
        self.action_space_type = Config.ACTION_SPACE_TYPE
        
        if self.action_space_type == Config.discrete:
            self.action_space = gym.spaces.Discrete(self.actions.num_actions, dtype=np.float32)
        elif self.action_space_type == Config.continuous:
            self.low_action = np.array([self.min_speed,
                                        self.min_heading_change])
            self.high_action = np.array([self.max_speed,
                                         self.max_heading_change])
            self.action_space = gym.spaces.Box(self.low_action, self.high_action, dtype=np.float32)
        

        # original observation space
        # self.observation_space = gym.spaces.Box(self.low_state, self.high_state, dtype=np.float32)
        
        # not used...
        # self.observation_space = np.array([gym.spaces.Box(self.low_state, self.high_state, dtype=np.float32)
                                           # for _ in range(self.num_agents)])
        # observation_space = gym.spaces.Box(self.low_state, self.high_state, dtype=np.float32)
        
        # single agent dict obs
        self.observation = {}
        for agent in range(Config.MAX_NUM_AGENTS_IN_ENVIRONMENT):
            self.observation[agent] = {}

        # The observation returned by the environment is a Dict of Boxes, keyed by agent index.
        self.observation_space = gym.spaces.Dict({})
        for state in Config.STATES_IN_OBS:
            self.observation_space.spaces[state] = gym.spaces.Box(Config.STATE_INFO_DICT[state]['bounds'][0]*np.ones((Config.STATE_INFO_DICT[state]['size'])),
                Config.STATE_INFO_DICT[state]['bounds'][1]*np.ones((Config.STATE_INFO_DICT[state]['size'])),
                dtype=Config.STATE_INFO_DICT[state]['dtype'])
            for agent in range(Config.MAX_NUM_AGENTS_IN_ENVIRONMENT):
                self.observation[agent][state] = np.zeros((Config.STATE_INFO_DICT[state]['size']), dtype=Config.STATE_INFO_DICT[state]['dtype'])

        self.agents = None
        self.default_agents = None
        self.prev_episode_agents = None

        self.static_map_filename = None
        self.map = None

        self.episode_step_number = None
        self.episode_number = 0

        self.plot_save_dir = None
        self.plot_policy_name = None

        self.perturbed_obs = None

    def step(self, actions, dt=None):
        """ Run one timestep of environment dynamics.

        This is the main function. An external process will compute an action for every agent
        then call env.step(actions). The agents take those actions,
        then we check if any agents have earned a reward (collision/goal/...).
        Then agents take an observation of the new world state. We compute whether each agent is done
        (collided/reached goal/ran out of time) and if everyone's done, the episode ends.
        We return the relevant info back to the process that called env.step(actions).

        Args:
            actions (list): list of [delta heading angle, speed] commands (1 per agent in env)
            dt (float): time in seconds to run the simulation (defaults to :code:`self.dt_nominal`)

        Returns:
        4-element tuple containing

        - **next_observations** (*np array*): (obs_length x num_agents) with each agent's observation
        - **rewards** (*list*): 1 scalar reward per agent in self.agents
        - **game_over** (*bool*): true if every agent is done
        - **info_dict** (*dict*): metadata that helps in training

        """

        if dt is None:
            dt = self.dt_nominal

        self.episode_step_number += 1

        # Take action
        self._take_action(actions, dt)

        # Collect rewards
        rewards = self._compute_rewards()

        # Take observation
        next_observations = self._get_obs()

        if Config.ANIMATE_EPISODES and self.episode_step_number % self.animation_period_steps == 0:
            plot_episode(self.agents, False, self.map, self.test_case_index,
                circles_along_traj=Config.PLOT_CIRCLES_ALONG_TRAJ,
                plot_save_dir=self.plot_save_dir,
                plot_policy_name=self.plot_policy_name,
                save_for_animation=True,
                limits=self.plt_limits,
                fig_size=self.plt_fig_size,
                perturbed_obs=self.perturbed_obs,
                show=False,
                save=True)

        # Check which agents' games are finished (at goal/collided/out of time)
        which_agents_done, game_over = self._check_which_agents_done()

        which_agents_done_dict = {}
        which_agents_learning_dict = {}
        for i, agent in enumerate(self.agents):
            which_agents_done_dict[agent.id] = which_agents_done[i]
            which_agents_learning_dict[agent.id] = agent.policy.is_still_learning

        return next_observations, rewards, game_over, \
            {
                'which_agents_done': which_agents_done_dict,
                'which_agents_learning': which_agents_learning_dict,
            }

    def reset(self):
        """ Resets the environment, re-initializes agents, plots episode (if applicable) and returns an initial observation.

        Returns:
            initial observation (np array): each agent's observation given the initial configuration
        """
        if self.episode_step_number is not None and self.episode_step_number > 0 and self.plot_episodes and self.test_case_index >= 0:
            plot_episode(self.agents, self.evaluate, self.map, self.test_case_index, self.id, circles_along_traj=Config.PLOT_CIRCLES_ALONG_TRAJ, plot_save_dir=self.plot_save_dir, plot_policy_name=self.plot_policy_name, limits=self.plt_limits, fig_size=self.plt_fig_size, show=Config.SHOW_EPISODE_PLOTS, save=Config.SAVE_EPISODE_PLOTS)
            if Config.ANIMATE_EPISODES:
                animate_episode(num_agents=len(self.agents), plot_save_dir=self.plot_save_dir, plot_policy_name=self.plot_policy_name, test_case_index=self.test_case_index, agents=self.agents)
            self.episode_number += 1
        self.begin_episode = True
        self.episode_step_number = 0
        self._init_agents()
        if Config.USE_STATIC_MAP:
            self._init_static_map()
        for state in Config.STATES_IN_OBS:
            for agent in range(Config.MAX_NUM_AGENTS_IN_ENVIRONMENT):
                self.observation[agent][state] = np.zeros((Config.STATE_INFO_DICT[state]['size']), dtype=Config.STATE_INFO_DICT[state]['dtype'])
        return self._get_obs()
    

    # # i add this def(For Finding points on a line with a given distance):
    # def get_point_on_vector(initial_pt, terminal_pt, distance):
    #     v = np.array(initial_pt, dtype=float)
    #     u = np.array(terminal_pt, dtype=float)
    #     n = v - u
    #     n /= np.linalg.norm(n, 2)
    #     point = v - distance * n
    #     return tuple(point)
    
    def _take_action(self, actions, dt):
        """ Some agents' actions come externally through the actions arg, agents with internal policies query their policy here, 
        then each agent takes a step simultaneously.

        This makes it so an external script that steps through the environment doesn't need to
        be aware of internals of the environment, like ensuring RVO agents compute their RVO actions.
        Instead, all policies that are already trained/frozen are computed internally, and if an
        agent's policy is still being trained, it's convenient to isolate the training code from the environment this way.
        Or, if there's a real robot with its own planner on-board (thus, the agent should have an ExternalPolicy), 
        we don't bother computing its next action here and just take what the actions dict said.

        Args:
            actions (dict): keyed by agent indices, each value has a [delta heading angle, speed] command.
                Agents with an ExternalPolicy sub-class receive their actions through this dict.
                Other agents' indices shouldn't appear in this dict, but will be ignored if so, because they have 
                an InternalPolicy sub-class, meaning they can
                compute their actions internally given their observation (e.g., already trained CADRL, RVO, Non-Cooperative, etc.)
            dt (float): time in seconds to run the simulation (defaults to :code:`self.dt_nominal`)

        """
        num_actions_per_agent = 2  # speed, delta heading angle
        all_actions = np.zeros((len(self.agents), num_actions_per_agent), dtype=np.float32)

        # Agents set their action (either from external or w/ find_next_action)
        for agent_index, agent in enumerate(self.agents):
            if agent.is_done:
                continue
            elif agent.policy.is_external:
                all_actions[agent_index, :] = agent.policy.external_action_to_action(agent, actions[agent_index])
            else:
                # # --------- Temp-Goal for RVO--------- (i add this)
                if Config.TmpGl:
                    if  agent.t >= 1 :                                  
                        stp0 = agent.global_state_history[agent.step_num-1][1:3]
                        stp1 = agent.global_state_history[agent.step_num-2][1:3]
                        stp2 = agent.global_state_history[agent.step_num-3][1:3]
                        stp3 = agent.global_state_history[agent.step_num-4][1:3]
                        stp4 = agent.global_state_history[agent.step_num-5][1:3]
                        stp5 = agent.global_state_history[agent.step_num-6][1:3]
                        stp6 = agent.global_state_history[agent.step_num-7][1:3]
                        stp7 = agent.global_state_history[agent.step_num-8][1:3]
                        stp8 = agent.global_state_history[agent.step_num-9][1:3]
                        # =============
                        shart = False #felan hazfesh kardam
                        
                        for ag_shrt_indx, agent_shrt in enumerate(self.agents):
                            # if ag_shrt_indx != agent_index and agent_shrt.is_done == False :
                            if ag_shrt_indx != agent_index :
                                dist2 = np.linalg.norm(agent.pos_global_frame-agent_shrt.pos_global_frame) - (agent.radius+agent_shrt.radius) # dist ag2ped-agRpd
                                if dist2 <= 2*Config.GETTING_CLOSE_RANGE :
                                    dist1 = np.linalg.norm(agent_shrt.pos_global_frame-agent.goal_global_frame) # dist ped2aggl
                                    # ----shrtWall--
                                    # if (tc.distped2orggl[ag_shrt_indx][1] != np.inf) and (tc.distped2orggl[ag_shrt_indx][1]!=ag_shrt_indx):
                                    #     if (np.linalg.norm(self.agents[ag_shrt_indx].pos_global_frame-self.agents[tc.distped2orggl[ag_shrt_indx][1]].pos_global_frame)-\
                                    #     (self.agents[ag_shrt_indx].radius+self.agents[tc.distped2orggl[ag_shrt_indx][1]].radius) )<= \
                                    #         (agent.radius*2):
                                    #         shrtWall = True
                                    # -------

                                    if agent_shrt.is_done == True: # baraye etminan az ist ped
                                        if ( ((agent.dist_to_goal-agent.radius) >= (dist1-agent_shrt.radius))or\
                                            (((agent.dist_to_goal-agent.radius) < (dist1-agent_shrt.radius)) and ((agent.dist_to_goal-agent.radius) <= dist2)))\
                                            and (dist2 <= agent.radius or dist2 <= agent_shrt.radius or dist2 <= 2*Config.GETTING_CLOSE_RANGE):
                                        # if ( ((agent.dist_to_goal-agent.radius) >= (dist1-agent_shrt.radius))or\
                                        #     (((agent.dist_to_goal-agent.radius) < (dist1-agent_shrt.radius)) and ((agent.dist_to_goal-agent.radius) <= dist2)) )\
                                        #     and (dist2 <= agent.radius or dist2 <= agent_shrt.radius or dist2 <= 2*Config.GETTING_CLOSE_RANGE):
                                                shart = True
                                                if tc.TG_sign[agent.id] == False and tc.cntnu_signS[agent.id] == True: # in shart baraye vaghtie k: 
                                                    # az stuck daromadim dar prose rafe stuck hastim 
                                                    # vali 1 ped dar masire rafe stuckman gharar darad va bayad on ro ham rad konim
                                                    dist3 = np.linalg.norm(agent.pos_global_frame-tc.OrgGl_pnt[agent.id]) #ag2orgGl
                                                    dist4 = np.linalg.norm(agent_shrt.pos_global_frame-tc.OrgGl_pnt[agent.id]) #ped2orgGl
                                                    tc.cntnu_sign[agent.id] = True
                                                    # if (( ((dist3-agent.radius) >= (dist4-agent_shrt.radius))or\
                                                    #     (((dist3-agent.radius) < (dist4-agent_shrt.radius)) and ((dist3-agent.radius) <= dist2)) )\
                                                    #         and (dist2 <= agent.radius or dist2 <= agent_shrt.radius or dist2 <= 2*Config.GETTING_CLOSE_RANGE))==False:
                                                    if ( ((dist3-agent.radius) >= (dist4-agent_shrt.radius)) and\
                                                         (dist2 <= agent.radius or dist2 <= agent_shrt.radius or dist2 <= 2*Config.GETTING_CLOSE_RANGE) )==False:
                                                        tc.cntnu_sign[agent.id]  = False
                                                        tc.cntnu_signS[agent.id] = False
                                        continue
                                    
                                    if ( ((agent.dist_to_goal-agent.radius) >= (dist1-agent_shrt.radius))or\
                                        (((agent.dist_to_goal-agent.radius) < (dist1-agent_shrt.radius)) and ((agent.dist_to_goal-agent.radius) <= dist2)) )\
                                        and (dist2 <= agent.radius or dist2 <= agent_shrt.radius or dist2 <= 2*Config.GETTING_CLOSE_RANGE):
                                        stp0_shrt = agent_shrt.global_state_history[agent.step_num-1][1:3]
                                        stp1_shrt = agent_shrt.global_state_history[agent.step_num-2][1:3]
                                        stp2_shrt = agent_shrt.global_state_history[agent.step_num-3][1:3]
                                        stp3_shrt = agent_shrt.global_state_history[agent.step_num-4][1:3]
                                        stp4_shrt = agent_shrt.global_state_history[agent.step_num-5][1:3]
                                        stp5_shrt = agent_shrt.global_state_history[agent.step_num-6][1:3]
                                        stp6_shrt = agent_shrt.global_state_history[agent.step_num-7][1:3]
                                        stp7_shrt = agent_shrt.global_state_history[agent.step_num-8][1:3]
                                        stp8_shrt = agent_shrt.global_state_history[agent.step_num-9][1:3]

                                        # sm_shrt = ((round(stp0_shrt[0],2) == round(stp1_shrt[0],2) == round(stp2_shrt[0],2) == round(stp3_shrt[0],2)
                                        #              == round(stp4_shrt[0],2) == round(stp5_shrt[0],2) == round(stp6_shrt[0],2) == round(stp7_shrt[0],2) == round(stp8_shrt[0],2))and\
                                        #            (round(stp0_shrt[1],2) == round(stp1_shrt[1],2) == round(stp2_shrt[1],2) == round(stp3_shrt[1],2)
                                        #              == round(stp4_shrt[1],2) == round(stp5_shrt[1],2) == round(stp6_shrt[1],2) == round(stp7_shrt[1],2) == round(stp8_shrt[1],2)))
                                        
                                        sm_shrt = (abs(round(stp0_shrt[0],2) - round(stp1_shrt[0],2)) <=0.011) and (abs(round(stp1_shrt[0],2) - round(stp2_shrt[0],2)) <=0.011) and \
                                        (abs(round(stp2_shrt[0],2) - round(stp3_shrt[0],2)) <=0.011) and (abs(round(stp3_shrt[0],2) - round(stp4_shrt[0],2)) <=0.011) and \
                                        (abs(round(stp4_shrt[0],2) - round(stp5_shrt[0],2)) <=0.011) and (abs(round(stp5_shrt[0],2) - round(stp6_shrt[0],2)) <=0.011) and \
                                        (abs(round(stp6_shrt[0],2) - round(stp7_shrt[0],2)) <=0.011) and (abs(round(stp7_shrt[0],2) - round(stp8_shrt[0],2)) <=0.011) and \
                                        (abs(round(stp0_shrt[1],2) - round(stp1_shrt[1],2)) <=0.011) and (abs(round(stp1_shrt[1],2) - round(stp2_shrt[1],2)) <=0.011) and \
                                        (abs(round(stp2_shrt[1],2) - round(stp3_shrt[1],2)) <=0.011) and (abs(round(stp3_shrt[1],2) - round(stp4_shrt[1],2)) <=0.011) and \
                                        (abs(round(stp4_shrt[1],2) - round(stp5_shrt[1],2)) <=0.011) and (abs(round(stp5_shrt[1],2) - round(stp6_shrt[1],2)) <=0.011) and \
                                        (abs(round(stp6_shrt[1],2) - round(stp7_shrt[1],2)) <=0.011) and (abs(round(stp7_shrt[1],2) - round(stp8_shrt[1],2)) <=0.011)
                                        # if (stp0_shrt[0] == stp1_shrt[0] == stp2_shrt[0] == stp3_shrt[0] == stp4_shrt[0] == stp5_shrt[0] == stp6_shrt[0] == stp7_shrt[0] == stp8_shrt[0] ) and\
                                        #     (stp0_shrt[1] == stp1_shrt[1] == stp2_shrt[1] == stp3_shrt[1] == stp4_shrt[1] == stp5_shrt[1] == stp6_shrt[1] == stp7_shrt[1] == stp8_shrt[1] ) and\
                                        #         (10*round(agent_shrt.global_state_history[agent.step_num-1][0],3)==agent.step_num-1):                                        
                                        if (sm_shrt) and (10*round(agent_shrt.global_state_history[agent.step_num-1][0],3)==agent.step_num-1):
                                            shart = True
                                            if tc.TG_sign[agent.id] == False and tc.cntnu_signS[agent.id] == True: # in shart baraye vaghtie k: 
                                                # az stuck daromadim dar prose rafe stuck  hastim 
                                                # vali 1 ped dar masire rafe stuckman gharar darad va bayad on ro ham rad konim
                                                dist3 = np.linalg.norm(agent.pos_global_frame-tc.OrgGl_pnt[agent.id])
                                                dist4 = np.linalg.norm(agent_shrt.pos_global_frame-tc.OrgGl_pnt[agent.id])
                                                tc.cntnu_sign[agent.id] = True
                                                # if (( ((dist3-agent.radius) >= (dist4-agent_shrt.radius))or\
                                                #      (((dist3-agent.radius) < (dist4-agent_shrt.radius)) and ((dist3-agent.radius) <= dist2)) )\
                                                #         and (dist2 <= agent.radius or dist2 <= agent_shrt.radius or dist2 <= 2*Config.GETTING_CLOSE_RANGE))==False:
                                                if ( ((dist3-agent.radius) >= (dist4-agent_shrt.radius)) and\
                                                     (dist2 <= agent.radius or dist2 <= agent_shrt.radius or dist2 <= 2*Config.GETTING_CLOSE_RANGE) )==False:
                                                    # hala nesbat b goal asli bbin aya nazdiktar az ped jadid hast ya na (agar pedi dar masireman nabashad)
                                                    tc.cntnu_sign[agent.id]  = False
                                                    tc.cntnu_signS[agent.id] = False

                        if shart == False:
                            tc.cntnu_sign[agent.id] = False
                        # =====================
                        # init for min and max side position of stuck zone pedestrian
                        min_ang1 = np.inf
                        max_ang1 = -np.inf
                        minpos_ang1 = np.empty
                        maxpos_ang1 = np.empty
                        # if self.test_case_index == 23: # bara rafe moshkel TG ORCA gozashtam
                        #     xxx = 1


                        shrtstkagnt = (abs(round(stp0[0],2) - round(stp1[0],2)) <=0.011) and (abs(round(stp1[0],2) - round(stp2[0],2)) <=0.011) and \
                                        (abs(round(stp2[0],2) - round(stp3[0],2)) <=0.011) and (abs(round(stp3[0],2) - round(stp4[0],2)) <=0.011) and \
                                        (abs(round(stp4[0],2) - round(stp5[0],2)) <=0.011) and (abs(round(stp5[0],2) - round(stp6[0],2)) <=0.011) and \
                                        (abs(round(stp6[0],2) - round(stp7[0],2)) <=0.011) and (abs(round(stp7[0],2) - round(stp8[0],2)) <=0.011) and \
                                        (abs(round(stp0[1],2) - round(stp1[1],2)) <=0.011) and (abs(round(stp1[1],2) - round(stp2[1],2)) <=0.011) and \
                                        (abs(round(stp2[1],2) - round(stp3[1],2)) <=0.011) and (abs(round(stp3[1],2) - round(stp4[1],2)) <=0.011) and \
                                        (abs(round(stp4[1],2) - round(stp5[1],2)) <=0.011) and (abs(round(stp5[1],2) - round(stp6[1],2)) <=0.011) and \
                                        (abs(round(stp6[1],2) - round(stp7[1],2)) <=0.011) and (abs(round(stp7[1],2) - round(stp8[1],2)) <=0.011)
                                    


                        
                        # == round(stp2[0],2) == round(stp3[0],2) == round(stp4[0],2)
                        #                  == round(stp5[0],2) == round(stp6[0],2) == round(stp7[0],2) == round(stp8[0],2))and\
                        #                (round(stp0[1],2) == round(stp1[1],2) == round(stp2[1],2) == round(stp3[1],2) == round(stp4[1],2)
                        #                  == round(stp5[1],2) == round(stp6[1],2) == round(stp7[1],2) == round(stp8[1],2)))
                        


                        # shrtstkagnt = ((round(stp0[0],2) == round(stp1[0],2) == round(stp2[0],2) == round(stp3[0],2) == round(stp4[0],2)
                        #                  == round(stp5[0],2) == round(stp6[0],2) == round(stp7[0],2) == round(stp8[0],2))and\
                        #                (round(stp0[1],2) == round(stp1[1],2) == round(stp2[1],2) == round(stp3[1],2) == round(stp4[1],2)
                        #                  == round(stp5[1],2) == round(stp6[1],2) == round(stp7[1],2) == round(stp8[1],2)))
                        # shrtstkagnt = ((stp0[0] == stp1[0] == stp2[0] == stp3[0] == stp4[0] == stp5[0] == stp6[0] == stp7[0] == stp8[0])and\
                        #                (stp0[1] == stp1[1] == stp2[1] == stp3[1] == stp4[1] == stp5[1] == stp6[1] == stp7[1] == stp8[1]))
                        if (shrtstkagnt or tc.cntnu_sign[agent.id] == True) and shart: # sign of stuck (albate in movaghatie ta bhtar peyda konim sign ro)
                        # if (shrtstkagnt) and shart: # sign of stuck (albate in movaghatie ta bhtar peyda konim sign ro)
                            TTdirection = False
                            if tc.TG_sign[agent.id] == False or (tc.cntnu_sign[agent.id] == True and shrtstkagnt==False):

                                tc.dist_ped2l90[agent.id] = 0 # chon harkat karde hadaghal 1 ghadam pas (ya avalin bare stuck shode) dist_ped2l90(mizan enheraf tempgoal(TT) nesbat b goal asli)
                                tc.TG_sign_agn[agent.id] = True
                                TTdirection = True
                                if tc.OrgGl_pnt[agent.id][0] != np.inf:
                                    self.agents[agent.id].goal_global_frame = tc.OrgGl_pnt[agent.id] # bargashte goal b goal asli (for what?!!)

                            # hala midonim stuck shode:
                            angle1 = ['nan' for _ in range(len(self.agents))]

                            # # init for min and max side position of stuck zone pedestrian

                            # -----------------------------------------------------------
                            for agent_index1, agent1 in enumerate(self.agents):
                                if agent_index != agent_index1:
                                    dist1 = np.linalg.norm(agent1.pos_global_frame-agent.goal_global_frame) # dist ped2aggl
                                    dist2 = np.linalg.norm(agent.pos_global_frame-agent1.pos_global_frame) - (agent.radius+agent1.radius) # dist ag2ped-agRpd
                                    # --shrtWall
                                    shrtWall = False
                                    # if (tc.distped2orggl[agent_index][1] != np.inf) and (tc.distped2orggl[agent_index][1]!=agent_index1):
                                    #     ddist1 = np.linalg.norm(self.agents[agent_index1].pos_global_frame-self.agents[tc.distped2orggl[agent_index][1]].pos_global_frame) # ped2besideped(Wall)
                                    #     radbsd1 = (self.agents[agent_index1].radius+self.agents[tc.distped2orggl[agent_index][1]].radius) #pedRbesideped
                                    #     if ((ddist1-radbsd1)<=(agent.radius*2))and\
                                    #     (dist2 < (ddist1-radbsd1)):
                                    #         shrtWall = True
                                    # ----
                                    if ( ((agent.dist_to_goal-agent.radius) >= (dist1-agent1.radius))or\
                                        (((agent.dist_to_goal-agent.radius) < (dist1-agent1.radius)) and ((agent.dist_to_goal-agent.radius) <= dist2))or\
                                        (shrtWall))and\
                                        (dist2 <= agent.radius or dist2 <= agent1.radius or dist2 <= 2*Config.GETTING_CLOSE_RANGE):# on ped morede nazare ma(samte chap va rast tarineshon ro peyda mikone) k baes stuck ma shode 
                                        # ino shayad bayad bzorgtar greft \:!?
                                        line_pd2ag = Line(Point(agent1.pos_global_frame),Point(agent.pos_global_frame))
                                        line_gl2ag = Line(Point(agent.goal_global_frame),Point(agent.pos_global_frame))

                                        line_for_sign = sign( ((agent.goal_global_frame[0]-agent.pos_global_frame[0])*(agent1.pos_global_frame[1]-agent.pos_global_frame[1]))-\
                                                             ((agent.goal_global_frame[1]-agent.pos_global_frame[1])*(agent1.pos_global_frame[0]-agent.pos_global_frame[0])) ) # Left or right agent2goal ههline
                                        angle1[agent_index1] = np.degrees(float(N((line_pd2ag).angle_between(line_gl2ag))))*(line_for_sign)
                                        
                                        # for min(rasttarinesh) and max(chap tarinesh) side position of stuck zone pedestrian
                                        if angle1[agent_index1] != 'nan': # niaz b in shart nabood! :)
                                            if angle1[agent_index1] <= min_ang1: # min angle yani samte raste agent mas
                                                min_ang1 = angle1[agent_index1]
                                                minpos_ang1 = agent_index1
                                            if angle1[agent_index1] >= max_ang1: # max angle yani samte chape agent mas
                                                max_ang1 = angle1[agent_index1]
                                                maxpos_ang1 = agent_index1

                            if minpos_ang1 != np.empty and maxpos_ang1 != np.empty:
                                tc.TG_sign[agent.id] = True
                                tc.cntnu_signS[agent.id] = True # sharte edame dadan
                                tc.ttstepsign[agent.id] = 0
                                    

                                if maxpos_ang1 == minpos_ang1: # in baraye vaghtie ke roye 1done pedesterian stuck shode
                                    if tc.OrgGl_pnt[agent.id][0] == np.inf:
                                        dist_ag2minped2gl = sign( ((self.agents[minpos_ang1].pos_global_frame[0]-agent.pos_global_frame[0])*(agent.goal_global_frame[1]-agent.pos_global_frame[1])) -\
                                                                    ((self.agents[minpos_ang1].pos_global_frame[1]-agent.pos_global_frame[1])*(agent.goal_global_frame[0]-agent.pos_global_frame[0])) )
                                        dist_ag2maxped2gl = dist_ag2minped2gl * (-1)
                                    else:
                                        dist_ag2minped2gl = sign( ((self.agents[minpos_ang1].pos_global_frame[0]-agent.pos_global_frame[0])*(tc.OrgGl_pnt[agent.id][1]-agent.pos_global_frame[1])) -\
                                                                    ((self.agents[minpos_ang1].pos_global_frame[1]-agent.pos_global_frame[1])*(tc.OrgGl_pnt[agent.id][0]-agent.pos_global_frame[0])) )
                                        dist_ag2maxped2gl = dist_ag2minped2gl * (-1)
                                    
                                else:
                                    dist_ag2minped2gl = np.linalg.norm(agent.pos_global_frame-self.agents[minpos_ang1].pos_global_frame) + \
                                        np.linalg.norm(self.agents[minpos_ang1].pos_global_frame-agent.goal_global_frame) + (2*self.agents[minpos_ang1].radius)
                                    dist_ag2maxped2gl = np.linalg.norm(agent.pos_global_frame-self.agents[maxpos_ang1].pos_global_frame) + \
                                        np.linalg.norm(self.agents[maxpos_ang1].pos_global_frame-agent.goal_global_frame) + (2*self.agents[maxpos_ang1].radius)

                                

                                if tc.dist_ped2l90[agent.id] >= 5:
                                    mn = dist_ag2minped2gl
                                    mx = dist_ag2maxped2gl
                                    dist_ag2minped2gl = mx
                                    dist_ag2maxped2gl = mn

                                    tc.dist_ped2l90[agent.id] = 0
                                    self.agents[agent.id].goal_global_frame = tc.OrgGl_pnt[agent.id] # bargashte goal b goal asli (for what?!!)


                                # inja yani akharin samte rastie agent entekhab shode(pas ma TT ro b samte rast mikshonim)
                                if (dist_ag2minped2gl <= dist_ag2maxped2gl and (TTdirection==False or tc.direction[agent.id]==0)) or (TTdirection and tc.direction[agent.id]==1) :
                                # if dist_ag2minped2gl <= dist_ag2maxped2gl  :
                                    agent_min = self.agents[minpos_ang1]
                                    sum_rds = self.agents[minpos_ang1].radius + agent.radius
                                    line_ag2pd = Line(Point(agent.pos_global_frame),Point(agent_min.pos_global_frame))
                                    Prp_ag2pd_pd = line_ag2pd.perpendicular_line(Point(agent_min.pos_global_frame)) # perpendicular agent2pdesterian line point on pedesterian
                                    Prp_ag2pd_pd_points = np.float64(np.array(Prp_ag2pd_pd.points[1])) 


                                    lforit = sign( ((agent_min.pos_global_frame[0]-agent.pos_global_frame[0])*(Prp_ag2pd_pd_points[1]-agent.pos_global_frame[1])) -\
                                                                ((agent_min.pos_global_frame[1]-agent.pos_global_frame[1])*(Prp_ag2pd_pd_points[0]-agent.pos_global_frame[0])) )

                                
                                    if lforit < 0: # noghte amod samte raste line agent2pedesterian
                                        dist_ped2l90 = sum_rds + Config.GETTING_CLOSE_RANGE+tc.dist_ped2l90[agent.id]
                                        
                                        pnt_ped2l90 = tc.get_point_on_vector(agent_min.pos_global_frame, Prp_ag2pd_pd_points, dist_ped2l90)  # b andaze dist_ped2l90 az agent_min b samte Prp_ag2pd_pd_points harkat mikonim
                                        dist_ag2TT = np.linalg.norm(agent.pos_global_frame-agent.goal_global_frame) 
                                        # pnt_ag2l90_2TT = tc.get_point_on_vector(agent.pos_global_frame, pnt_ped2l90, dist_ag2TT) # this is Temporary goal
                                        pnt_ag2l90_2TT = tc.get_point_on_vector(agent.pos_global_frame, pnt_ped2l90, 3*agent.pref_speed) # saat3 sobh 21om
                                    elif lforit > 0:
                                        dist_ped2l90 = sum_rds + Config.GETTING_CLOSE_RANGE + np.linalg.norm(Prp_ag2pd_pd_points-agent_min.pos_global_frame)+tc.dist_ped2l90[agent.id]
                                        pnt_ped2l90 = tc.get_point_on_vector(Prp_ag2pd_pd_points, agent_min.pos_global_frame, dist_ped2l90)
                                        dist_ag2TT = np.linalg.norm(agent.pos_global_frame-agent.goal_global_frame)
                                        # pnt_ag2l90_2TT = tc.get_point_on_vector(agent.pos_global_frame, pnt_ped2l90, dist_ag2TT) # this is Temporary goal
                                        pnt_ag2l90_2TT = tc.get_point_on_vector(agent.pos_global_frame, pnt_ped2l90, 3*agent.pref_speed) # saat3 sobh 21om

                                    

                                    TT =  np.float64(pnt_ag2l90_2TT) # TT: temporary target

                                    dist_ag2gl = np.linalg.norm(agent.pos_global_frame-tc.OrgGl_pnt[agent.id])
                                    distTT=dist_ag2gl
                                    # distTT=2*agent.pref_speed
                                    # while tc.TG_sign[agent.id] == False:
                                    sgn_all_agnt = 0
                                    while sgn_all_agnt == len(self.agents)-1:
                                        sgn_all_agnt = 0
                                        for i in range(len(self.agents)):
                                            if i != agent.id:
                                                if round(np.linalg.norm(TT - self.agents[i].pos_global_frame),3) >= \
                                                    round((agent.radius + self.agents[i].radius + Config.GETTING_CLOSE_RANGE),3): # yani agar TT point ma ba agenti dar tasadom nist
                                                    # tc.TG_sign[agent.id] = True
                                                    sgn_all_agnt = sgn_all_agnt+1
                                                else:
                                                    dstpd2agttl = Line(Point(agent.pos_global_frame),Point(TT)).distance(Point(self.agents[i].pos_global_frame))
                                                    distTT = (agent.radius + self.agents[i].radius + Config.GETTING_CLOSE_RANGE)**2 - dstpd2agttl**2
                                                    # tc.TG_sign[agent.id] = False
                                                    # distTT = distTT+(np.linalg.norm(TT - self.agents[i].pos_global_frame) - (agent.radius + self.agents[i].radius) + Config.GETTING_CLOSE_RANGE)
                                                    distag2pd = np.linalg.norm(self.agents[i].pos_global_frame - agent.pos_global_frame)
                                                    distTT = distTT+distag2pd**2 - dstpd2agttl**2
                                                    TT = np.float64(tc.get_point_on_vector(agent.pos_global_frame, pnt_ped2l90, distTT)) # this is Temporary goal


                                    tc.angl_ag2pdbtwag2ag[agent.id] = 30 # movaghati nerkh sabet

                                    tc.distped2aggl[agent.id] = np.linalg.norm(TT - agent_min.pos_global_frame)


                                # inja yani akharin samte chapie agent entekhab shode
                                elif (dist_ag2minped2gl > dist_ag2maxped2gl and (TTdirection==False or tc.direction[agent.id]==0)) or (TTdirection and tc.direction[agent.id]==2):
                                # elif dist_ag2minped2gl > dist_ag2maxped2gl:
                                    agent_max = self.agents[maxpos_ang1]
                                    sum_rds = self.agents[maxpos_ang1].radius + agent.radius
                                    line_ag2pd = Line(Point(agent.pos_global_frame),Point(agent_max.pos_global_frame))
                                    Prp_ag2pd_pd = line_ag2pd.perpendicular_line(Point(agent_max.pos_global_frame)) # perpendicular agent2pdesterian line point on pedesterian
                                    Prp_ag2pd_pd_points = np.float64(np.array(Prp_ag2pd_pd.points[1])) 


                                    lforit = sign( ((agent_max.pos_global_frame[0]-agent.pos_global_frame[0])*(Prp_ag2pd_pd_points[1]-agent.pos_global_frame[1])) -\
                                                                ((agent_max.pos_global_frame[1]-agent.pos_global_frame[1])*(Prp_ag2pd_pd_points[0]-agent.pos_global_frame[0])) )
                                    if lforit > 0: # noghte amod samte chape line agent2pedesterian
                                        dist_ped2l90 = sum_rds + Config.GETTING_CLOSE_RANGE+tc.dist_ped2l90[agent.id]
                                        pnt_ped2l90 = tc.get_point_on_vector(agent_max.pos_global_frame, Prp_ag2pd_pd_points, dist_ped2l90)
                                        dist_ag2TT = np.linalg.norm(agent.pos_global_frame-agent.goal_global_frame) 
                                        # pnt_ag2l90_2TT = tc.get_point_on_vector(agent.pos_global_frame, pnt_ped2l90, dist_ag2TT) # this is Temporary goal
                                        pnt_ag2l90_2TT = tc.get_point_on_vector(agent.pos_global_frame, pnt_ped2l90, 3*agent.pref_speed) # saat 3sobh
                                        
                                    elif lforit < 0:
                                        dist_ped2l90 = sum_rds + Config.GETTING_CLOSE_RANGE + np.linalg.norm(Prp_ag2pd_pd_points-agent_max.pos_global_frame)+tc.dist_ped2l90[agent.id]
                                        pnt_ped2l90 = tc.get_point_on_vector(Prp_ag2pd_pd_points, agent_max.pos_global_frame, dist_ped2l90) 
                                        dist_ag2TT = np.linalg.norm(agent.pos_global_frame-agent.goal_global_frame) 
                                        # pnt_ag2l90_2TT = tc.get_point_on_vector(agent.pos_global_frame, pnt_ped2l90, dist_ag2TT) # this is Temporary goal
                                        pnt_ag2l90_2TT = tc.get_point_on_vector(agent.pos_global_frame, pnt_ped2l90, 3*agent.pref_speed) # this is Temporary goal

                                    
                                    TT =  np.float64(pnt_ag2l90_2TT) # TT: temporary target
                                    # if np.inf == tc.firstTTpnt[agent.id][0]:
                                    #     tc.firstTTpnt[agent.id] = TT # save first TT point in tc.firstTTpnt
                                    dist_ag2gl = np.linalg.norm(agent.pos_global_frame-tc.OrgGl_pnt[agent.id])
                                    distTT=dist_ag2gl
                                    # distTT=2*agent.pref_speed
                                    # while tc.TG_sign[agent.id] == False:
                                    sgn_all_agnt = 0
                                    while sgn_all_agnt == len(self.agents)-1:
                                        sgn_all_agnt = 0
                                        for i in range(len(self.agents)):
                                            if i != agent.id:
                                                if round(np.linalg.norm(TT - self.agents[i].pos_global_frame),3) >= \
                                                     round((agent.radius + self.agents[i].radius + Config.GETTING_CLOSE_RANGE),3):
                                                    # TT =  pnt_ag2l90_2TT# TT: temporary target
                                                    # tc.TG_sign[agent.id] = True
                                                    sgn_all_agnt = sgn_all_agnt+1
                                                else:
                                                    dstpd2agttl = Line(Point(agent.pos_global_frame),Point(TT)).distance(Point(self.agents[i].pos_global_frame))
                                                    distTT = (agent.radius + self.agents[i].radius + Config.GETTING_CLOSE_RANGE)**2 - dstpd2agttl**2
                                                    # tc.TG_sign[agent.id] = False
                                                    # distTT = distTT+(np.linalg.norm(TT - self.agents[i].pos_global_frame) - (agent.radius + self.agents[i].radius) + Config.GETTING_CLOSE_RANGE)
                                                    distag2pd = np.linalg.norm(self.agents[i].pos_global_frame - agent.pos_global_frame)
                                                    distTT = distTT+distag2pd**2 - dstpd2agttl**2
                                                    TT = np.float64(tc.get_point_on_vector(agent.pos_global_frame, pnt_ped2l90, distTT)) # this is Temporary goal
                                    tc.angl_ag2pdbtwag2ag[agent.id] = 30 # felan ba nerkh sabet kam beshe

                                    tc.distped2aggl[agent.id] = np.linalg.norm(TT - agent_max.pos_global_frame)

                                
                                tc.dist_ped2l90[agent.id] = dist_ped2l90
                                tc.TmpGl_pnt[agent.id] = TT # add TT to TmpGl
                                if np.inf == tc.OrgGl_pnt[agent.id][0]:
                                    tc.OrgGl_pnt[agent.id] = agent.goal_global_frame # save orginal goal in tc.OrgGl_pnt
                                    tc.distagpos2agt = np.linalg.norm(agent.pos_global_frame-agent.goal_global_frame)
                                if (dist_ag2minped2gl <= dist_ag2maxped2gl and (TTdirection==False or tc.direction[agent.id]==0)) or (TTdirection and tc.direction[agent.id]==1):
                                    tc.distped2orggl[agent.id] = (np.linalg.norm(tc.OrgGl_pnt[agent.id] - agent_min.pos_global_frame),agent_min.id)
                                    if lforit < 0:
                                        tc.direction[agent.id] = 1
                                    else:
                                        tc.direction[agent.id] = 1

                                elif (dist_ag2minped2gl > dist_ag2maxped2gl and (TTdirection==False or tc.direction[agent.id]==0)) or (TTdirection and tc.direction[agent.id]==2):
                                    tc.distped2orggl[agent.id] = (np.linalg.norm(tc.OrgGl_pnt[agent.id] - agent_max.pos_global_frame),agent_max.id)
                                    if lforit > 0:
                                        tc.direction[agent.id] = 2
                                    else:
                                        tc.direction[agent.id] = 2

                                self.agents[agent.id].goal_global_frame  = TT # add Temporary Target(goal) to agent goal
                                

                        # if (self.default_agents[1].t) >= 12.1:
                        #     xxx=1
                            #--- in ghesmate code darhale kam kardane zavie TT b goal hast ta dar nahayat b khode goal bresim -------------
                        if ((tc.TG_sign[agent.id] == True or tc.TG_sign_agn[agent.id]==True) and tc.distped2aggl[agent.id]!=np.inf and \
                            ( np.linalg.norm(agent.pos_global_frame-agent.goal_global_frame) != np.linalg.norm(agent.global_state_history[agent.step_num-2][1:3]-agent.goal_global_frame) )):
                            tc.TG_sign[agent.id] = False
                            # new RVO
                            tc.distped2orggl[agent.id] = (np.linalg.norm(tc.OrgGl_pnt[agent.id] - self.agents[tc.distped2orggl[agent.id][1]].pos_global_frame), tc.distped2orggl[agent.id][1])
                            # 
                            if (np.linalg.norm(agent.pos_global_frame-agent.goal_global_frame) <= tc.distped2aggl[agent.id]-(agent.pref_speed/2))\
                                or (tc.cntnu_sign[agent.id]==False and (np.linalg.norm(agent.pos_global_frame-tc.OrgGl_pnt[agent.id]) <= tc.distped2orggl[agent.id][0]\
                                                                        or ((agent.goal_global_frame == tc.OrgGl_pnt[agent.id])[0] and
                                                                             (agent.goal_global_frame == tc.OrgGl_pnt[agent.id])[1]) )):
                                
                                # tc.firstTTpnt[agent.id] = [np.inf,np.inf] #new idea 2
                                tc.dist_ped2l90[agent.id] = 0 # new idea
                                # tc.TG_sign[agent.id] = False
                                line_ag2TT = Line(Point(agent.pos_global_frame),Point(tc.TmpGl_pnt[agent.id]))
                                line_ag2gl = Line(Point(agent.pos_global_frame),Point(tc.OrgGl_pnt[agent.id]))
                                if (np.degrees(float(N((line_ag2TT).angle_between(line_ag2gl)))) <= tc.angl_ag2pdbtwag2ag[agent.id]) or tc.ttstepsign[agent.id]==4 :
                                    self.agents[agent.id].goal_global_frame = tc.OrgGl_pnt[agent.id]
                                    tc.TG_sign_agn[agent.id]=False
                                    # tc.TG_sign[agent.id] = False
                                    # tc.dist_ped2l90[agent.id] = 0 # new idea
                                    tc.OrgGl_pnt[agent.id] = [np.inf,np.inf]
                                    tc.cntnu_signS[agent.id] = False
                                    tc.cntnu_sign[agent.id] = False
                                    tc.direction[agent.id] = 0
                                    tc.ttstepsign[agent.id] = 0
                                else:
                                    best_sign = False # chon hanoz behtarin zavie baraye kam kardan TT b goal asli peyda nashode
                                    bestTT = tc.OrgGl_pnt[agent.id] # hamon TT

                                    if (np.degrees(float(N((line_ag2TT).angle_between(line_ag2gl)))) > 90) or tc.ttstepsign[agent.id] == (3): #agar bish az 90 darage bashad
                                        best_sign = True
                                        tc.ttstepsign[agent.id] += 1
                                        lnag2l902gl = line_ag2gl.perpendicular_line(Point(agent.pos_global_frame))
                                        ag2l902glpnt = np.float64(np.array(lnag2l902gl.points[1]))
                                        lforit90 = sign( ((bestTT[0]-agent.pos_global_frame[0])*(ag2l902glpnt[1]-agent.pos_global_frame[1])) -\
                                                                    ((bestTT[1]-agent.pos_global_frame[1])*(ag2l902glpnt[0]-agent.pos_global_frame[0])) )
                                        lforitTT = sign( ((bestTT[0]-agent.pos_global_frame[0])*(tc.TmpGl_pnt[agent.id]-agent.pos_global_frame[1])) -\
                                                                    ((bestTT[1]-agent.pos_global_frame[1])*(tc.TmpGl_pnt[agent.id]-agent.pos_global_frame[0])) )
                                        if lforit90 == lforitTT: # noghte amod samte chape line agent2pedesterian
                                            bestTT = ag2l902glpnt
                                        else:
                                            dist902ag = np.linalg.norm(agent.pos_global_frame-ag2l902glpnt) 
                                            bestTT = tc.get_point_on_vector(ag2l902glpnt , agent.pos_global_frame, 2*dist902ag)
                                    

                                    prebest_sign = False
                                    while best_sign == False: # ta vaghti peyda nashode edame bede
                                        prebestTT = bestTT
                                        bestTT = np.float64(np.array(Point(bestTT).midpoint(Point(tc.TmpGl_pnt[agent.id]))))
                                        line_ag2best_sign = Line(Point(agent.pos_global_frame),Point(bestTT))
                                        
                                        if (np.degrees(float(N((line_ag2TT).angle_between(line_ag2best_sign)))) <= 30):
                                            best_sign = True
                                            tc.ttstepsign[agent.id] += 1

                                    bestTT = tc.get_point_on_vector(agent.pos_global_frame , bestTT, 2*agent.pref_speed) # new idea baraye fasele k mikhaym
                                    dist_ag2gl = np.linalg.norm(agent.pos_global_frame-tc.OrgGl_pnt[agent.id])
                                    distTT=dist_ag2gl
                                    TT =  np.float64(bestTT)
                                    sgn_all_agnt = 0
                                    while sgn_all_agnt == len(self.agents)-1:
                                        sgn_all_agnt = 0
                                        for i in range(len(self.agents)):
                                            if i != agent.id:
                                                if round(np.linalg.norm(TT - self.agents[i].pos_global_frame),3) >= \
                                                    round((agent.radius + self.agents[i].radius + Config.GETTING_CLOSE_RANGE),3):
                                                    sgn_all_agnt = sgn_all_agnt+1
                                                else:
                                                    dstpd2agttl = Line(Point(agent.pos_global_frame),Point(TT)).distance(Point(self.agents[i].pos_global_frame))
                                                    distTT = (agent.radius + self.agents[i].radius + Config.GETTING_CLOSE_RANGE)**2 - dstpd2agttl**2
                                                    distag2pd = np.linalg.norm(self.agents[i].pos_global_frame - agent.pos_global_frame)
                                                    distTT = distTT+distag2pd**2 - dstpd2agttl**2
                                                    TT = np.float64(tc.get_point_on_vector(agent.pos_global_frame, bestTT, distTT)) # this is Temporary goal

                                    tc.TmpGl_pnt[agent.id] = TT # add TT to TmpGl
                                    self.agents[agent.id].goal_global_frame  = TT
                                    
                # --------- end TPG ----------
                
                dict_obs = self.observation[agent_index]
                all_actions[agent_index, :] = agent.policy.find_next_action(dict_obs, self.agents, agent_index)

        # After all agents have selected actions, run one dynamics update
        for i, agent in enumerate(self.agents):
            agent.take_action(all_actions[i,:], dt)

    def _update_top_down_map(self):
        """ After agents have moved, call this to update the map with their new occupancies. """
        self.map.add_agents_to_map(self.agents)
        # plt.imshow(self.map.map)
        # plt.pause(0.1)

    def set_agents(self, agents):
        """ Set the default agent configuration, which will get used at the start of each episode (and bypass calling self.test_case_fn)

        Args:
            agents (list): of :class:`~gym_collision_avoidance.envs.agent.Agent` objects that should become the self.default_agents
                and thus be loaded in that configuration every time the env resets.

        """
        self.default_agents = agents

    def _init_agents(self):
        """ Set self.agents (presumably at the start of a new episode) and set each agent's max heading change and speed based on env limits.

        self.agents gets set to self.default_agents if it exists.
        Otherwise, self.agents gets set to the result of self.test_case_fn(self.test_case_args).        
        """

        # The evaluation scripts need info about the previous episode's agents
        # (in case env.reset was called and thus self.agents was wiped)
        if self.evaluate and self.agents is not None:
            self.prev_episode_agents = copy.deepcopy(self.agents)

        # If nobody set self.default agents, query the test_case_fn
        if self.default_agents is None:
            self.agents = self.test_case_fn(**self.test_case_args)
        # Otherwise, somebody must want the agents to be reset in a certain way already
        else:
            self.agents = self.default_agents

        # Make every agent respect the same env-wide limits on actions (this probably should live elsewhere...)
        for agent in self.agents:
            agent.max_heading_change = self.max_heading_change
            agent.max_speed = self.max_speed

    def set_static_map(self, map_filename):
        """ If you want to have static obstacles, provide the path to the map image file that should be loaded.
        
        Args:
            map_filename (str or list): full path of a binary png file corresponding to the environment prior map 
                (or list of candidate map paths to randomly choose btwn each episode)
        """
        self.static_map_filename = map_filename

    def _init_static_map(self):
        """ Load the map based on its pre-provided filename, and initialize a :class:`~gym_collision_avoidance.envs.Map.Map` object

        Currently the dimensions of the world map are hard-coded.

        """
        if isinstance(self.static_map_filename, list):
            static_map_filename = np.random.choice(self.static_map_filename)
        else:
            static_map_filename = self.static_map_filename

        x_width = 16 # meters
        y_width = 16 # meters
        grid_cell_size = 0.1 # meters/grid cell
        self.map = Map(x_width, y_width, grid_cell_size, static_map_filename)

    def _compute_rewards(self):
        """ Check for collisions and reaching of the goal here, and also assign the corresponding rewards based on those calculations.
        
        Returns:
            rewards (scalar or list): is a scalar if we are only training on a single agent, or
                      is a list of scalars if we are training on mult agents
        """

        # if nothing noteworthy happened in that timestep, reward = -0.01
        rewards = self.reward_time_step*np.ones(len(self.agents))
        collision_with_agent, collision_with_wall, entered_norm_zone, dist_btwn_nearest_agent = \
            self._check_for_collisions()

        for i, agent in enumerate(self.agents):
            if agent.is_at_goal:
                if agent.was_at_goal_already is False:
                    # agents should only receive the goal reward once
                    rewards[i] = self.reward_at_goal
                    # print("Agent %i: Arrived at goal!"
                          # % agent.id)
            else:
                # agents at their goal shouldn't be penalized if someone else
                # bumps into them
                if agent.was_in_collision_already is False:
                    if collision_with_agent[i]:
                        rewards[i] = self.reward_collision_with_agent
                        agent.in_collision = True
                        # print("Agent %i: Collision with another agent!"
                        #       % agent.id)
                    elif collision_with_wall[i]:
                        rewards[i] = self.reward_collision_with_wall
                        agent.in_collision = True
                        # print("Agent %i: Collision with wall!"
                              # % agent.id)
                    else:
                        # There was no collision
                        if dist_btwn_nearest_agent[i] <= Config.GETTING_CLOSE_RANGE:
                            rewards[i] = -0.1 - dist_btwn_nearest_agent[i]/2.
                            # print("Agent %i: Got close to another agent!"
                            #       % agent.id)
                        if abs(agent.past_actions[0, 1]) > self.wiggly_behavior_threshold:
                            # Slightly penalize wiggly behavior
                            rewards[i] += self.reward_wiggly_behavior
                        # elif entered_norm_zone[i]:
                        #     rewards[i] = self.reward_entered_norm_zone
        rewards = np.clip(rewards, self.min_possible_reward,
                          self.max_possible_reward)
        if Config.TRAIN_SINGLE_AGENT:
            rewards = rewards[0]
        return rewards

    def _check_for_collisions(self):
        """ Check whether each agent has collided with another agent or a static obstacle in the map 
        
        This method doesn't compute social zones currently!!!!!

        Returns:
            - collision_with_agent (list): for each agent, bool True if that agent is in collision with another agent
            - collision_with_wall (list): for each agent, bool True if that agent is in collision with object in map
            - entered_norm_zone (list): for each agent, bool True if that agent entered another agent's social zone
            - dist_btwn_nearest_agent (list): for each agent, float closest distance to another agent

        """
        collision_with_agent = [False for _ in self.agents]
        collision_with_wall = [False for _ in self.agents]
        entered_norm_zone = [False for _ in self.agents]
        dist_btwn_nearest_agent = [np.inf for _ in self.agents]
        agent_shapes = []
        agent_front_zones = []
        agent_inds = list(range(len(self.agents)))
        agent_pairs = list(itertools.combinations(agent_inds, 2))
        for i, j in agent_pairs:
            dist_btwn = l2norm(self.agents[i].pos_global_frame, self.agents[j].pos_global_frame)
            combined_radius = self.agents[i].radius + self.agents[j].radius
            dist_btwn_nearest_agent[i] = min(dist_btwn_nearest_agent[i], dist_btwn - combined_radius)
            dist_btwn_nearest_agent[j] = min(dist_btwn_nearest_agent[j], dist_btwn - combined_radius)
            if dist_btwn <= combined_radius:
                # Collision with another agent!
                collision_with_agent[i] = True
                collision_with_agent[j] = True
        if Config.USE_STATIC_MAP:
            for i in agent_inds:
                agent = self.agents[i]
                [pi, pj], in_map = self.map.world_coordinates_to_map_indices(agent.pos_global_frame)
                mask = self.map.get_agent_map_indices([pi, pj], agent.radius)
                # plt.figure('static map')
                # plt.imshow(self.map.static_map + mask)
                # plt.pause(0.1)
                if in_map and np.any(self.map.static_map[mask]):
                    # Collision with wall!
                    collision_with_wall[i] = True
        return collision_with_agent, collision_with_wall, entered_norm_zone, dist_btwn_nearest_agent

    def _check_which_agents_done(self):
        """ Check if any agents have reached goal, run out of time, or collided.

        Returns:
            - which_agents_done (list): for each agent, True if agent is done, o.w. False
            - game_over (bool): depending on mode, True if all agents done, True if 1st agent done, True if all learning agents done
        """
        at_goal_condition = np.array(
                [a.is_at_goal for a in self.agents])
        ran_out_of_time_condition = np.array(
                [a.ran_out_of_time for a in self.agents])
        in_collision_condition = np.array(
                [a.in_collision for a in self.agents])
        which_agents_done = np.logical_or.reduce((at_goal_condition, ran_out_of_time_condition, in_collision_condition))
        for agent_index, agent in enumerate(self.agents):
            agent.is_done = which_agents_done[agent_index]
        
        if Config.EVALUATE_MODE:
            # Episode ends when every agent is done
            game_over = np.all(which_agents_done)
        elif Config.TRAIN_SINGLE_AGENT:
            # Episode ends when ego agent is done
            game_over = which_agents_done[0]
        else:
            # Episode is done when all *learning* agents are done
            learning_agent_inds = [i for i in range(len(self.agents)) if self.agents[i].policy.is_still_learning]
            game_over = np.all(which_agents_done[learning_agent_inds])
        
        return which_agents_done, game_over

    def _get_obs(self):
        """ Update the map now that agents have moved, have each agent sense the world, and fill in their observations 

        Returns:
            observation (list): for each agent, a dictionary observation.

        """

        if Config.USE_STATIC_MAP:
            # Agents have moved (states have changed), so update the map view
            self._update_top_down_map()

        # Agents collect a reading from their map-based sensors
        for i, agent in enumerate(self.agents):
            agent.sense(self.agents, i, self.map)

        # Agents fill in their element of the multiagent observation vector
        for i, agent in enumerate(self.agents):
            self.observation[i] = agent.get_observation_dict(self.agents)

        return self.observation

    def _initialize_rewards(self):
        """ Set some class attributes regarding reward values based on Config """
        self.reward_at_goal = Config.REWARD_AT_GOAL
        self.reward_collision_with_agent = Config.REWARD_COLLISION_WITH_AGENT
        self.reward_collision_with_wall = Config.REWARD_COLLISION_WITH_WALL
        self.reward_getting_close = Config.REWARD_GETTING_CLOSE
        self.reward_entered_norm_zone = Config.REWARD_ENTERED_NORM_ZONE
        self.reward_time_step = Config.REWARD_TIME_STEP

        self.reward_wiggly_behavior = Config.REWARD_WIGGLY_BEHAVIOR
        self.wiggly_behavior_threshold = Config.WIGGLY_BEHAVIOR_THRESHOLD

        self.possible_reward_values = \
            np.array([self.reward_at_goal,
                      self.reward_collision_with_agent,
                      self.reward_time_step,
                      self.reward_collision_with_wall,
                      self.reward_wiggly_behavior
                      ])
        self.min_possible_reward = np.min(self.possible_reward_values)
        self.max_possible_reward = np.max(self.possible_reward_values)

    def set_plot_save_dir(self, plot_save_dir):
        """ Set where to save plots of trajectories (will get created if non-existent)
        
        Args:
            plot_save_dir (str): path to directory you'd like to save plots in

        """
        makedirs(plot_save_dir, exist_ok=True)
        self.plot_save_dir = plot_save_dir

    def set_perturbed_info(self, perturbed_obs):
        """ Used for robustness paper to pass info that could be visualized. Too hacky.
        """
        self.perturbed_obs = perturbed_obs

    def set_testcase(self, test_case_fn_str, test_case_args):
        """ 

        Args:
            test_case_fn_str (str): name of function in test_cases.py
        """

        # Provide a fn (which returns list of agents) and the fn's args,
        # to be called on each env.reset()
        test_case_fn = getattr(tc, test_case_fn_str, None)
        assert(callable(test_case_fn))

        # Before running test_case_fn, make sure we didn't provide any args it doesn't accept
        if sys.version[0] == '3':
            signature = inspect.signature(test_case_fn)
        elif sys.version[0] == '2':
            import funcsigs
            signature = funcsigs.signature(test_case_fn)
        test_case_fn_args = signature.parameters
        test_case_args_keys = list(test_case_args.keys())
        for key in test_case_args_keys:
            # print("checking if {} accepts {}".format(test_case_fn, key))
            if key not in test_case_fn_args:
                # print("{} doesn't accept {} -- removing".format(test_case_fn, key))
                del test_case_args[key]
        self.test_case_fn = test_case_fn
        self.test_case_args = test_case_args

if __name__ == '__main__':
    print("See example.py for a minimum working example.")
