import blackjack_base as bj
import gym.spaces as spaces
import numpy as np
from gym.utils import seeding
from math import inf


class BlackjackEnvExtend(bj.BlackjackEnvBase):
    """
    Class which extends OpenAI BlackJackEnv class such that it is a proper
    stationary Markov decision process.

    Observation space is expanded, the agent now sees the number of cards
    it is holding at each state.
    """
    def __init__(self, decks = inf, seed=3232, natural=True):
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Tuple((
        # MultiDiscrete is a vector of the number of possible values per element
                spaces.MultiDiscrete([22,11,8,6,5,4,4,3,3,3]),
                spaces.Discrete(26)))
        self.seed(seed)
        # initialize the number of cards to have of each deck
        # Flag to payout 1.5 on a "natural" blackjack win, like casino rules
        # Ref: http://www.bicyclecards.com/how-to-play/blackjack/
        self.natural = natural
        self.decks = decks # number of decks
        # Start the first game
        self.reset()

    def is_natural(self):
        # A hand is a natural blackjack if it has 2 cars which total 21
        return sum(self.player) == 2 and self.sum_player_hand() == 21

    def is_player_bust(self):
        return self.sum_player_hand() > 21

    def score_player(self):
        return 0 if self.is_player_bust() else self.sum_player_hand()

    def sum_player_hand(self):
        return np.dot(self.deck_values, self.player) + \
                10 * self.usable_player_ace()

    def usable_player_ace(self):
        return self.player[0] > 0 and \
                np.dot(self.deck_values, self.player) + 10 <= 21

    def draw_player_card(self):
        self.player[self.draw_card() - 1] +=1

    def draw_player_hand(self):
        hand = np.zeros(len(self.deck_values), int)
        hand[self.draw_card() - 1] += 1
        hand[self.draw_card() - 1] += 1
        return hand

    def _get_obs(self):
        return (tuple(self.player), self.dealer_show_cards())

    def reset(self):
        self.done = False
        self.construct_deck()
        self.dealer = self.draw_hand()
        self.player = self.draw_player_hand()
        return self._get_obs()

