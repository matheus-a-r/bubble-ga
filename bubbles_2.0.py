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
		play(background, guns, game, grid_manager, networks, genomes, i)
	
	return

def sum_color(bubble):
	if not bubble.exists:
		return -1.0
	
	color_sum = bubble.color[0] + bubble.color[1] + bubble.color[2]

	if color_sum == 256:
		color_sum = 0
	elif color_sum == 257:
		color_sum = 1
	elif color_sum == 258:
		color_sum = 2
	elif color_sum == 420:
		color_sum = 3
	elif color_sum == 510:
		color_sum = 4
	elif color_sum == 382: 
		color_sum = 5

	return color_sum


def play(background, guns, game, grid_manager, networks, genomes_list, i):
	background.draw()						
	while not game.over:
		grid_manager.view(guns[i], game)

		old_score = game.score
		
		inputs = []
		for j in range(len(grid_manager.grid)):
			if j < 3:
				for bubble in grid_manager.grid[j]:
					inputs.append(sum_color(bubble))

		inputs.append(guns[i].loaded.color[0] + guns[i].loaded.color[1] + guns[i].loaded.color[2])

		res = networks[i].activate(inputs)
		target = (res[0] * DISP_W, res[1] * DISP_H)
		
		guns[i].rotate(target)
		guns[i].fire()
		new_score = game.score
		if new_score > old_score:
			genomes_list[i].fitness += 0.1

		guns[i].draw_bullets()			

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


