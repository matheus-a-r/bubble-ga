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
		gun = Shooter(pos = BOTTOM_CENTER)
		gun.putInBox()
		
		network = neat.nn.FeedForwardNetwork.create(genome, config)
		networks.append(network)
		genome.fitness = 0
		genomes_list.append(genome)
		guns.append(gun)

	# Create background	

	background = Background()

	for gun in guns:
		game = Game()
		
		grid_manager = GridManager()
		play(background, gun, game, grid_manager)
	
	return

def play(background, gun, game, grid_manager):
	
	background.draw()				# Draw BG first		
	while not game.over:
		grid_manager.view(gun, game)	# Check collision with bullet and update grid as needed		

		trigger_color = gun.loaded.color

		for i in grid_manager.targets: 				# para cada i nos alvos possíveis
			if i.color == trigger_color:
				for j in i.getComrades():
					if j.color == i.color:
						gun.rotate((i.pos[0], i.pos[1]))
									
								
		gun.fire()

		gun.draw_bullets()				# Draw and update bullet and reloads	

		game.drawScore()				# draw score

		pg.display.update()		
		clock.tick(60)

	game.gameOverScreen(grid_manager, background)	


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


