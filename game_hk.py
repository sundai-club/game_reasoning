import random
import os
from abc import ABC, abstractmethod
from vllm import LLM, SamplingParams
import ray

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'

@ray.remote
class LLMWrapper:

    def __init__(self):
        self.model = LLM(
            model='Qwen/Qwen2.5-1.5B-Instruct',
            gpu_memory_utilization=0.95, 
            max_model_len=2048,
            tensor_parallel_size=1, 
            enable_prefix_caching=True
        )
        self.sampling_params = SamplingParams(
            max_tokens=2000,
            n=1,  # Change for best of 256 eval
            temperature=0.7
        )
        self.queue_length = 1
        self.queue_idx = 0
        self.job_queue = [None for _ in range(self.queue_length)]
        self.results = []

    def set_prompt(self, prompt):
        idx = self.queue_idx
        self.job_queue[idx] = prompt
        return ray.get(self.get_response.remote(idx))

    def get_response(self, idx):
        while self.queue_idx < self.queue_idx:
            pass
        if idx == self.queue_idx:
            self.results = self.model.generate(
                self.job_queue,
                self.sampling_params,
                use_tqdm=True
            )
        return self.results[idx]
        
        
llm_wrapper = LLMWrapper.remote()

class Player:
    def __init__(self, name, dice_count=5):
        self.name = name
        self.dice = [0] * dice_count

    @property
    def dice_count(self):
        return len(self.dice)
    
    def face_count(self, face):
        return self.dice.count(face)

    def roll(self):
        self.dice = [random.randint(1, 6) for _ in self.dice]

    def lose_die(self):
        if self.dice:
            self.dice.pop()
            
    def get_action(self, *args):
        return Call()
        
class Action(ABC):
    
    @abstractmethod
    def __str__(self):
        pass
            
class Bid(Action):
    def __init__(self, quantity, face):
        self.quantity = quantity
        self.face = face

    def __str__(self):
        return f"{self.quantity} × {self.face}'s"
    
    def gt(self, other):
        assert isinstance(other, Bid) and self.is_valid() and other.is_valid()
        if self.quantity > other.quantity:
            return 1
        elif self.quantity < other.quantity:
            return -1
        else:
            return 1 if self.face > other.face else -1
        
    def is_valid(self):
        return 1 <= self.quantity <= 5 and 1 <= self.face <= 6
        
class Call(Action):
    def __init__(self):
        pass

    def __str__(self):
        return "Call"

class LiarDiceGame:
    def __init__(self, players):
        self.players = players
        self.num_players = len(self.players)
        self.current_bid = None  # tuple (quantity, face)
        self.bid_history = []
        self.turn = None

    def reset_round(self):
        self.turn = random.randrange(len(self.players))
        self.bid_history = []
        self.current_bid = None
        for p in self.players:
            p.roll()
        
    def prev_player(self, idx):
        return (idx - 1) % len(self.players)

    def next_player(self, idx):
        return (idx + 1) % len(self.players)

    def valid_bid(self, new_bid):
        return self.current_bid is None or (new_bid.is_valid() and new_bid.gt(self.current_bid))

    def count_face(self, face):
        return sum(p.face_count(face) for p in self.players)
    
    def wins_challenge(self):
        '''Resolve the challenge between the caller and the bidder.'''
        if self.current_bid is None:
            return False
        total = self.count_face(self.current_bid.face)
        if total >= self.current_bid.quantity:
            return False
        return True
            
    def eliminate_players(self):
        self.players = [p for p in self.players if p.dice_count > 0]

    def play_round(self):
        self.reset_round()
        while True:
            player = self.players[self.turn]
            action = player.get_action(self)
            if isinstance(action, Bid):
                if self.valid_bid(action):
                    self.current_bid = action
                    self.bid_history.append((player, action))
                else:
                    player.lose_die()
                    print(f"{player.name} made an invalid bid!")
                    break
            elif isinstance(action, Call):
                if self.wins_challenge():
                    bidder_idx = self.prev_player(self.turn)
                    bidder = self.players[bidder_idx]
                    bidder.lose_die()
                    print(f"{player.name} called the bid correctly!")
                    break
                else:
                    player.lose_die()
                    print(f"{player.name} called the bid incorrectly!")
                    break
            else:
                player.lose_die()
                print(f"{player.name} made an invalid action!")
                break
        self.eliminate_players()
        
        
class LLMPlayer(Player):
    def __init__(self, name, dice_count=5):
        super().__init__(name, dice_count)
        self.client = llm_wrapper

    def generate_response(self, data):
        history, num_players = data
        prompt = f"""You are playing a game of Liar's Dice. 
            Goal: The purpose of the game is to be the last player remaining with at least one die. 
            You lose a die each time you're caught bluffing—or if you wrongly challenge someone else's truthful bid.
            
            Rules: Each player gets 5 six-sided dice and a cup (to hide their roll). Everyone rolls their dice secretly and keeps their roll hidden under the cup.
            Players take turns making bids about how many dice (across all players) show a certain face value. 
            For example, “There are at least five 3s” means across everyone’s dice, the player is claiming there are five or more dice showing 3.

            Turn Structure:
            Each turn, you can either: Raise the bid (make a higher or riskier claim), or Challenge the previous player by calling them a liar.
            You must increase the bid by: Quantity: “Six 3s” → “Seven 3s” OR Face value: “Six 3s” → “Six 4s” OR Both: “Six 3s” → “Seven 4s”. Note: You can’t go backwards in value or quantity.
            If you think the previous bid is a lie, say “Liar!”. 
            Everyone reveals their dice. Count how many dice show the face that was bid.
            If the bid was true:Challenger loses a die.
            If the bid was false: Bidder loses a die.
            Then everyone rerolls their dice and a new round begins.

            End condition: Players are eliminated when they lose all their dice. The game continues until one player remains.

            Strategy:
            - Bluff early, when many dice are in play—harder to call.
            - Track your own dice, use them to guess probabilities.
            - Adapt—if you have fewer dice, bluffing gets riskier.

            Game Conditions: You have the following dice: {self.dice}.
            There are {num_players} players in the game.
            The bid history is:
            """
        if history:
            for player, action in history:
                prompt += f"{player.name} bids {action}\n"
        else:
            prompt += "No bids yet.\n"
            
        prompt += f"""Your turn. What action do you want to take? (Bid or Call)
            You can bid a quantity of dice and a face value (1-6), or call the last player's bid.
            Remember, you can only call if you think the last player is lying.
            If you call and the last player is telling the truth, you lose a die.
            If you bid and the next player calls, you lose a die if your bid is incorrect.
            Enclose your action in a box.
            Enclose your thinking inside <think> </think> tags.
            Verify your process and make sure your action is valid.
            Your action should be in the format:
            <action>Bid(quantity, face)</action> or <action>Call</action>.
        """
        response = self.client.set_prompt.remote(prompt)
        return response
    
    def parse_response(self, response):
        try:
            action = response.split("<action>")[1].split("</action>")[0]
            if "Bid" in action:
                quantity, face = map(int, action[4:-1].split(","))
                return Bid(quantity, face)
            elif "Call" in action:
                return Call()
            else:
                # Could not parse action!
                return Bid(-1, -1)
        except Exception as e:
            print(f"Error parsing response: {response}")
            # If parsing fails, return an invalid bid
            # to ensure the player loses a die.
            return Bid(-1, -1)

    def get_action(self, game_data):
        response = self.generate_response([game_data.bid_history, game_data.num_players])
        action = self.parse_response(response)
        return action
    
    def get_reasoning(self, response):
        try:
            reasoning = response.split("<think>")[1].split("</think>")[0].strip()
            return reasoning
        except Exception as e:
            print(f"Error extracting thinking from response: {response}")
            return "No thinking available"

@ray.remote
def run_game(num_players):
    # build a fresh game inside the remote task
    players = [LLMPlayer(f"Player {i}") for i in range(num_players)]
    game = LiarDiceGame(players)

    while len(game.players) > 1:
        game.play_round()
        print(f"Round over. Remaining players: {[p.name for p in game.players]}")

    winner = game.players[0].name
    print(f"{winner} wins the game!")
    return winner

def main():
    PARALLEL_GAMES = 1
    futures = [run_game.remote(2) for _ in range(PARALLEL_GAMES)]
    winners = ray.get(futures)
    print("All games finished. Winners:", winners)

if __name__ == "__main__":
    main()