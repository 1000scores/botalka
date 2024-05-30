
import random
from tqdm import tqdm

epochs = 10000
N = int(1e4)

d = 0
k = 0
noone = 0
for epocn in tqdm(range(epochs)):
    player_Denis = 1000000
    player_Katya = 1000000
    start_d_bet = 50
    d_bet = start_d_bet
    
    winner = None
    for i in range(N):
        s = f'{i}. '
        k_choice = random.randint(0, 1)
        d_choice = False
        coin = random.randint(0, 1)
        k_bet = random.randint(500, 2000)
        
        if player_Katya < k_bet or player_Denis < d_bet:
            if player_Katya < k_bet and player_Denis < d_bet:
                winner = 'Noone'
            elif player_Katya < k_bet:
                winner = 'Denis'
                #print('Katya LOST')
            elif player_Denis < d_bet:
                winner = 'Katya'
                #print('Denis LOST')
            break

        if k_choice == coin:
            player_Katya += k_bet
            s += f'Katya win: {k_bet}. Balance: {player_Katya} | '
        else:
            player_Katya -= k_bet
            s += f'Katya lost: {k_bet}. Balance: {player_Katya} | '
        
        if d_choice == coin:
            player_Denis += d_bet
            s += f'Denis win: {d_bet}. Balance: {player_Denis}'
            d_bet = start_d_bet
        else:
            player_Denis -= d_bet
            s += f'Denis lost: {d_bet}. Balance: {player_Denis} | '
            d_bet = d_bet * 2  
        #print(s)
    if winner is None:
        if player_Denis == player_Katya:
            noone += 1
        elif player_Denis > player_Katya:
            d += 1
        else:
            k += 1
    else:
        if winner == 'Noone':
            noone += 1
        elif winner == 'Denis':
            d += 1
        else:
            k += 1
            
print(f'Denis won {100. * d / epochs}% times. Katya won {100. * k / epochs}% times. Noone {1. * noone / epochs}% times')
    
    