import gym
import gym.spaces as spaces
from gym.utils import seeding
import numpy as np
"""
This code extends the BlackjackEnv from openAI gym to a finite deck.
"""


def cmp(a, b):
    return float(a > b) - float(a < b)


class BlackjackEnvBase(gym.Env):
    """Simple blackjack environment

    Blackjack is a card game where the goal is to obtain cards that sum to as
    near as possible to 21 without going over.  They're playing against a fixed
    dealer.
    Face cards (Jack, Queen, King) have point value 10.
    Aces can either count as 11 or 1, and it's called 'usable' at 11.
    This game is placed with an infinite deck (or with replacement).
    The game starts with each (player and dealer) having one face up and one
    face down card.

    The player can request additional cards (hit=1) until they decide to stop
    (stick=0) or exceed 21 (bust).

    After the player sticks, the dealer reveals their facedown card, and draws
    until their sum is 17 or greater.  If the dealer goes bust the player wins.

    If neither player nor dealer busts, the outcome (win, lose, draw) is
    decided by whose sum is closer to 21.  The reward for winning is +1,
    drawing is 0, and losing is -1.

    The observation of a 3-tuple of: the players current sum,
    the dealer's one showing card (1-10 where 1 is ace),
    and whether or not the player holds a usable ace (0 or 1).

    This environment corresponds to the version of the blackjack problem
    described in Example 5.1 in Reinforcement Learning: An Introduction
    by Sutton and Barto (1998).
    http://incompleteideas.net/sutton/book/the-book.html
    """
    # 1 = Ace, 2-10 = Number cards, Jack/Queen/King = 10
    # deck = [1,2,3,4,5,6,7,8,9,10,10,10,10]
    deck_values = [x for x in range(1, 11)]

    def __init__(self, decks, seed, natural=True):
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(32),
            spaces.Discrete(26),
            spaces.Discrete(2)))
        self.seed(seed)

        # Flag to payout 1.5 on a "natural" blackjack win, like casino rules
        # Ref: http://www.bicyclecards.com/how-to-play/blackjack/
        self.natural = natural
        self.decks = decks # number of decks
        # Start the first game
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)
        if action:  # hit: add a card to players hand and return
            self.draw_player_card()
            if self.is_player_bust():
                self.done = True
                reward = -1
            else:
                self.done = False
                reward = 0
        else:  # stick: play out the dealers hand, and score
            self.done = True
            while self.dealer_show_cards() < 17:
                self.dealer.append(self.draw_card())
            reward = self.calculate_reward()
        return self._get_obs(), reward, self.done, {}

    def calculate_reward(self):
        # returns the score of the table.
        if self.natural:
            if self.is_natural() and len(self.dealer) > 2:
                return 1.5
        return cmp(self.score_player(), self.score_dealer())

    def score(self, hand):  # What is the score of this hand (0 if bust)
        return 0 if self.is_bust(hand) else self.sum_hand(hand)

    def score_player(self):
        return self.score(self.player)

    def score_dealer(self):
        return self.score(self.dealer)

    def is_natural(self):
        return sorted(self.player) == [1, 10]

    def is_bust(self, hand):
        return True if self.sum_hand(hand) > 21 else False

    def is_player_bust(self):
        return self.is_bust(self.player)

    def sum_hand(self, hand):
        return sum(hand)+ 10 * self.usable_ace(hand)

    def usable_ace(self, hand):  # Does this hand have a usable ace?
        return 1 in hand and sum(hand) + 10 <= 21

    def draw_player_card(self):
        self.player.append(self.draw_card())

    def construct_deck(self):
        self.cards_in_deck = {x: self.decks for x in self.deck_values}
        # since we are looking at deck_values: 10, knight, queen, king
        # are valued equally. Update the last element such that we have 4 times
        # as many cards
        self.cards_in_deck[10] = self.decks * 4

    def subtract_card_from_deck(self, card):
        #print(self.cards_in_deck[card])
        if self.cards_in_deck[card] > 1:
            # if there is more than one card left, subtract it!
            self.cards_in_deck[card] -= 1
        else:
            # if there is exactly one card left, than after it is used we
            # remove the key, thus we cannot draw the card again
            self.cards_in_deck.pop(card)

    def draw_card(self):
        # we can only draw cards which are in the keys of cards_in_deck.
        if(np.isinf(self.decks)):
            weights = [1,1,1,1,1,1,1,1,1,4]
        else:
            weights = np.fromiter(self.cards_in_deck.values(), dtype=float)
        probs = weights/np.sum(weights)
        card = int(self.np_random.choice(list(self.cards_in_deck.keys()),p=probs))
        #print(card)
        # subtract the card from the deck
        self.subtract_card_from_deck(card)
        return card

    def draw_hand(self):
        return [self.draw_card(), self.draw_card()]

    def reset(self):
        self.done = False
        self.construct_deck()
        self.dealer = self.draw_hand()
        self.player = self.draw_hand()
        return self._get_obs()

    def dealer_show_cards(self):
        if self.done:
            return self.sum_hand(self.dealer)
        else:
            return self.dealer[0]

    def _get_obs(self):
        return (self.sum_hand(self.player), self.dealer_show_cards(),
                self.usable_ace(self.player))

