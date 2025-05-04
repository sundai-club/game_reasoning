import random
import os
from abc import ABC, abstractmethod
# from vllm import LLM, SamplingParams
import asyncio
from enum import Enum

import openai
from openai import OpenAI
import dotenv
import json

import ray

dotenv.load_dotenv()

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'

ray.init(
    num_cpus=96
)

@ray.remote(num_cpus=0)
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
            # max_tokens=2000,
            n=1,  # Change for best of 256 eval
            temperature=0.7
        )
        self.queue_length = 64
        self.queue_idx = 0
        self.queue_barrier = asyncio.Barrier(self.queue_length)
        self.job_queue = [None for _ in range(self.queue_length)]
        self.responses = [None for _ in range(self.queue_length)]
        
    async def set_prompt(self, prompt):
        idx = self.queue_idx
        self.job_queue[idx] = prompt
        self.queue_idx += 1
        # The last request in the queue will trigger the model to generate
        if idx == self.queue_length - 1:
            self.responses = self.model.generate(
                self.job_queue,
                sampling_params=self.sampling_params,
                max_concurrent_requests=self.queue_length
            )
            self.queue_idx = 0
        self.queue_barrier.wait()
        return self.responses[idx]
    
# llm_wrapper = LLMWrapper.remote()
    
@ray.remote(num_cpus=32)
class APIWrapper:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE"),
        )
    
    def set_prompt(self, prompt):
        response = self.client.chat.completions.create(
            messages=[
                {"role": "user", "content": prompt}
            ],
            model='Qwen/Qwen3-235B-A22B-fp8-tput',
            temperature=0.7,
            # max_tokens=2000,
            n=1
        )
        return response.choices[0].message.content
        
api_wrapper = APIWrapper.remote()

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
    
    def __str__(self):
        return f"{self.name}"
        
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
        return new_bid.is_valid() and (self.current_bid is None or new_bid.gt(self.current_bid))

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
        self.loser = None
        self.winner = None
        while True:
            player = self.players[self.turn]
            action = player.get_action(self)
            self.bid_history.append((player, action))
            if isinstance(action, Bid):
                if self.valid_bid(action):
                    self.current_bid = action
                else:
                    player.lose_die()
                    self.loser = player
                    print(f"{player.name} made an invalid bid!")
                    break
            elif isinstance(action, Call):
                bidder_idx = self.prev_player(self.turn)
                bidder = self.players[bidder_idx]
                if self.wins_challenge():
                    bidder.lose_die()
                    self.winner = player
                    self.loser = bidder
                    print(f"{player.name} called the bid correctly!")
                    break
                else:
                    player.lose_die()
                    self.winner = bidder
                    self.loser = player
                    print(f"{player.name} called the bid incorrectly!")
                    break
            else:
                player.lose_die()
                self.loser = player
                print(f"{player.name} made an invalid action!")
                break
            self.turn = self.next_player(self.turn)
        # Logging the round results
        self.eliminate_players()
        if self.winner:
            self.winner.commit()
        for p in self.players:
            p.clear()
        
        
class LLMPlayer(Player):
    
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
            print(f"Error parsing response")
            # If parsing fails, return an invalid bid
            # to ensure the player loses a die.
            return Bid(-1, -1)

    def get_action(self, game_data):
        response = self.generate_response([game_data.bid_history, game_data.num_players])
        action = self.parse_response(response)
        return action
    
    def generate_response(self, data):
        raise NotImplementedError("Subclasses should implement this method.")
    
        
class vLLMPlayer(LLMPlayer):
    def __init__(self, name, dice_count=5):
        super().__init__(name, dice_count)
        self.client = llm_wrapper
        
    def generate_prompt(self, data):
        history, num_players = data
        prompt = f"""
            /no_think
            You are playing a game of Liar's Dice. 
            Goal: The purpose of the game is to be the last player remaining with at least one die.

            Rules: Each player gets 5 six-sided dice and are hidden from other players. Everyone rolls their dice secretly and keeps their roll hidden.
            Players take turns making bids about how many dice (across all players) show a certain face value. 
            For example, If the current bid is “Bid(5, 3)” means across everyone’s dice, the player is claiming there are five or more dice showing 3.

            Turn Structure:
            Each turn, you can either: Raise the bid, or challenge and call the previous player bid.
            If you are raising the bid you must increase the bid either by quantity or face value or both for eg: Quantity: Current Bid - (6, 3) → “Bid (7, 3)” OR Face value: Current Bid - (6, 3) → “Bid(6, 4)” OR Both: Current Bid - (6, 3) → “Bid(7, 4)”. Note: You can’t go backwards in value or quantity.
            If you want to call the previous bid, say “Call”. 
            Everyone reveals their dice and the current face are counted.
            If the actual count of that face value among all dice is less than the bid, the last bidder loses one die.
            Otherwise, the caller loses one die.
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
            Verify your process and make sure your action is valid.
            Your action should be in the format:
            <action>Bid(quantity, face)</action> or <action>Call</action>. /no_think
        """
        return prompt

    def generate_response(self, data):
        prompt = self.generate_prompt(data)
        response_future = asyncio.run(self.client.set_prompt(prompt))
        return ray.get(response_future)
    
class APIPlayer(LLMPlayer):
    
    def __init__(self, name, dice_count=5):
        super().__init__(name, dice_count)
        self.client = api_wrapper
    
    def generate_prompt(self, data):
        history, num_players = data
        prompt = f"""
            /no_think
            Think step by step, but be concise in your reasoning. Focus on the most important factors and avoid unnecessary details.
            Keep your thinking under 400 words.
            
            You are playing a game of Liar's Dice. 
            Goal: The purpose of the game is to be the last player remaining with at least one die.

            Rules: Each player gets 5 six-sided dice and are hidden from other players. Everyone rolls their dice secretly and keeps their roll hidden.
            Players take turns making bids about how many dice (across all players) show a certain face value. 
            For example, If the current bid is “Bid(5, 3)” means across everyone’s dice, the player is claiming there are five or more dice showing 3.

            Turn Structure:
            Each turn, you can either: Raise the bid, or challenge and call the previous player bid.
            If you are raising the bid you must increase the bid either by quantity or face value or both for eg: Quantity: Current Bid - (6, 3) → “Bid (7, 3)” OR Face value: Current Bid - (6, 3) → “Bid(6, 4)” OR Both: Current Bid - (6, 3) → “Bid(7, 4)”. Note: You can’t go backwards in value or quantity.
            If you want to call the previous bid, say “Call”. 
            Everyone reveals their dice and the current face are counted.
            If the actual count of that face value among all dice is less than the bid, the last bidder loses one die.
            Otherwise, the caller loses one die.
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
            Verify your process and make sure your action is valid.
            Your action should be in the format:
            <action>Bid(quantity, face)</action> or <action>Call</action>. /no_think
        """
        return prompt
    
    def generate_response(self, data):
        prompt = self.generate_prompt(data)
        return ray.get(self.client.set_prompt.remote(prompt))
        
# Replace the original LogResponseWrapper selection with this wrapper‐around‐APIPlayer
class LogResponseWrapper:
    def __init__(self, player):
        # wrap an APIPlayer instance
        self.player = player
        self.prompt_queue = []
        self.reasoning_queue = []

    @property
    def name(self):
        return self.player.name

    @property
    def dice_count(self):
        return self.player.dice_count

    @property
    def dice(self):
        return self.player.dice

    @dice.setter
    def dice(self, new_dice):
        self.player.dice = new_dice

    def roll(self):
        return self.player.roll()

    def lose_die(self):
        return self.player.lose_die()

    def face_count(self, face):
        return self.player.face_count(face)

    def get_action(self, game):
        # Prepare the prompt & call the wrapped APIPlayer
        data = [game.bid_history, game.num_players]
        prompt = self.player.generate_prompt(data)
        response = ray.get(self.player.client.set_prompt.remote(prompt))
        
        # extract and log reasoning
        reasoning = self._extract_reason(response)
        self.prompt_queue.append(prompt)
        self.reasoning_queue.append(reasoning)
        
        # parse & return the action
        return self.player.parse_response(response)

    def _extract_reason(self, response):
        # try:
        #     return response.split("<think>")[1].split("</think>")[0].strip()
        # except Exception:
        #     # If parsing fails, return the entire response as reasoning
        #     return response
        return response

    def commit(self):
        # write prompt, reasoning pairs to a CSV
        fn = f"{self.player.name}_log.jsonl"
        with open(fn, "a") as f:
            for prompt, reasoning in zip(self.prompt_queue, self.reasoning_queue):
                entry = {
                    "messages": [
                        {"role": "user",      "content": prompt},
                        {"role": "assistant", "content": reasoning}
                    ]
                }
                f.write(json.dumps(entry) + "\n")

    def clear(self):
        self.prompt_queue.clear()
        self.reasoning_queue.clear()
    
@ray.remote(num_cpus=32)
def run_game(num_players):
    # build a fresh game inside the remote task
    players = [LogResponseWrapper(APIPlayer(f"Player {i}")) for i in range(num_players)]
    game = LiarDiceGame(players)

    while len(game.players) > 1:
        game.play_round()
        print(f"Round over. Remaining players: {[p.name for p in game.players]}")

    winner = game.players[0].name
    print(f"{winner} wins the game!")
    return winner

def main():
    PARALLEL_GAMES = 1024
    NUM_PLAYERS = 2
    futures = [run_game.remote(NUM_PLAYERS) for _ in range(PARALLEL_GAMES)]
    winners = ray.get(futures)
    print("All games finished. Winners:", winners)

if __name__ == "__main__":
    main()