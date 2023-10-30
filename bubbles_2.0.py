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
	mouse_pos = (DISP_W/2, DISP_H/2)
	background = Background()

	
	for i, gun in enumerate(guns):
		game = Game()
		
		grid_manager = GridManager()
		play(background, gun, game, grid_manager, mouse_pos, i, guns, network, genomes_list)
	
	return

def play(background, gun, game, grid_manager, mouse_pos, i, guns, network, genomes_list):
	
	background.draw()						
	while not game.over:
		grid_manager.view(gun, game)			

		old_score = game.score
		x, y = network[i].activate((gun.loaded.color,
                                        grid_manager.targets,
                                    ))
		target = (x * DISP_W, y * DISP_H)
		
		gun.rotate(target)
		gun.fire()
		new_score = game.score
		if new_score > old_score:
			genomes_list[i].fitness += 0.1

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


