import time, random
from objs.constants import *
from objs.bubble_file import *
from objs.grid_file import *
from objs.shooter_file import *
from objs.game_objects import *
import pygame as pg
import neat
import os

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
		gun = Shooter(pos = BOTTOM_CENTER)
		gun.putInBox()
		
		network = neat.nn.FeedForwardNetwork.create(genome, config)
		networks.append(network)
		genome.fitness = 0
		genomes_list.append(genome)
		guns.append(gun)

	# Create background	

	# Starting mouse position
	background = Background()

	
	for i in range(len(genomes)):
		game = Game()
		
		grid_manager = GridManager()
		play(background, guns[i], game, grid_manager, networks[i], genomes_list[i])
	
	return

def sum_color(bubble):
	if not bubble.exists:
		return -1.0

	return bubble.color[0] + bubble.color[1] + bubble.color[2]


def play(background, gun, game, grid_manager, network, genomes_list):
	background.draw()						
	while not game.over:
		grid_manager.view(gun, game)

		old_score = game.score
		
		inputs = []
		for i in range(len(grid_manager.grid)):
			if i < 3:
				for bubble in grid_manager.grid[i]:
					inputs.append(sum_color(bubble))

		inputs.append(gun.loaded.color[0] + gun.loaded.color[1] + gun.loaded.color[2])

		res = network.activate(inputs)
		target = (res[0] * DISP_W, res[1] * DISP_H)
		
		gun.rotate(target)
		gun.fire()
		new_score = game.score
		if new_score > old_score:
			genomes_list.fitness += 0.1

		gun.draw_bullets()					

		game.drawScore()				

		pg.display.update()		
		clock.tick(60)

	#game.gameOverScreen(grid_manager, background)	


def run(path_config):
	config = neat.config.Config(neat.DefaultGenome,
								neat.DefaultReproduction,
								neat.DefaultSpeciesSet,
								neat.DefaultStagnation,
								path_config)

	population = neat.Population(config)
	population.add_reporter(neat.StdOutReporter(True))
	population.add_reporter(neat.StatisticsReporter())
	population.run(main, 50)

if __name__ == '__main__': 
	path = os.path.dirname(__file__)
	path_config = os.path.join(path, 'config.txt')
	run(path_config)


