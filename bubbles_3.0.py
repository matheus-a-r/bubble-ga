import time, random
from objs.constants import *
from objs.bubble_file import *
from objs.grid_file import *
from objs.shooter_file import *
from objs.game_objects import *
import pygame as pg
import neat
import os
import time

pg.init()

generation = 0

ia_play = True

def main(genomes, config):
    global generation, n
    generation += 1

    networks = []
    genomes_list = []
    guns = []
    
    for _, genome in genomes:
        gun = Shooter(pos=BOTTOM_CENTER)
        gun.putInBox()
        
        # Configurar a rede neural com base nas opções do arquivo de configuração
        config_net = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                 neat.DefaultSpeciesSet, neat.DefaultStagnation, config)
        network = neat.nn.FeedForwardNetwork.create(genome, config_net)
        
        networks.append(network)
        genome.fitness = 0
        genomes_list.append(genome)
        guns.append(gun)

    # Create background
    background = Background()

    for gun in guns:
        game = Game()
        
        grid_manager = GridManager()
        play(background, gun, game, grid_manager, networks)  # Adicione networks como argumento

    return

def play(background, gun, game, grid_manager, networks):
    
    background.draw()  # Draw BG first        
    while not game.over:
        grid_manager.view(gun, game)
        
        trigger_color = gun.loaded.color
        for i in grid_manager.targets:
            if i.color == trigger_color:
                # Verificar se a bola "target" tem uma vizinha de mesma cor
                if has_same_color_neighbor(i, grid_manager.targets):
                    # Usar a rede neural para tomar ação com base na configuração
                    action = get_network_action(networks[gun.ID], gun, grid_manager, game)
                    update_shooter(gun, action)
        
        gun.fire()
        gun.draw_bullets()
        game.drawScore()
        
        pg.display.update()
        time.sleep(1)  # Adicione um atraso (ajuste conforme necessário)
        clock.tick(60)

    game.gameOverScreen(grid_manager, background)

def has_same_color_neighbor(target, targets):
    for comrade in target.getComrades():
        if comrade.color == target.color:
            return True
    return False

def get_network_action(network, gun, grid_manager, game):
    # Defina como você deseja que a rede neural calcule suas saídas com base no ambiente e na configuração
    inputs = get_inputs(gun, grid_manager, game)  # Substitua pela lógica de coleta de dados de entrada
    outputs = network.activate(inputs)
    
    # Decodifique as saídas da rede neural e retorne as ações
    return decode_outputs(outputs)

def get_inputs(gun, grid_manager, game):
    # Colete os dados de entrada com base no ambiente, como a posição do "shooter", bolas "target" etc.
    # Certifique-se de que a estrutura de entrada corresponda à configuração da rede neural
    inputs = [gun, grid_manager, game]  # Substitua pelos valores reais das entradas
    return inputs

def decode_outputs(outputs):
    # Exemplo simples: suponha que você tenha duas saídas, onde a primeira indica procurar e a segunda indica atirar
    search_action = outputs[0]  # Primeira saída
    shoot_action = outputs[1]  # Segunda saída

    # Decodificar as saídas com base em algum critério (por exemplo, um valor acima de um limite)
    if search_action > 0.5:
        search = True
    else:
        search = False

    if shoot_action > 0.5:
        shoot = True
    else:
        shoot = False

    return search, shoot

# Função para atualizar o "shooter" com base nas ações
def update_shooter(shooter, game, grid_manager, actions):
    search, shoot = actions

    if search:
        # Procurar por bolas "target" com mesma cor e pelo menos uma vizinha de mesma cor
        target = find_target_with_neighbors(shooter.loaded.color, grid_manager.targets)

        if target is not None:
            # Se encontrar uma bola "target" válida, ajustar a mira (rotate)
            shooter.rotate((target.pos[0], target.pos[1]))

    if shoot:
        shooter.fire()

# Função para calcular a aptidão com base no desempenho do "shooter"
def calculate_fitness(shooter, game):
    # Exemplo simples: aptidão com base na pontuação do jogo
    fitness = game.score

    return fitness

# Função para encontrar uma bola "target" com mesma cor e pelo menos uma vizinha de mesma cor
def find_target_with_neighbors(color, targets):
    for target in targets:
        if target.color == color:
            comrades = target.getComrades()

            for comrade in comrades:
                if comrade.color == color:
                    return target  # Encontrou um alvo válido

    return None  # Nenhum alvo válido encontrado