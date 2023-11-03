import time, random
from objs.constants import *
from objs.bubble_file import *
from objs.grid_file import *
from objs.shooter_file import *
from objs.game_objects import *
import pygame as pg
import neat
import os
import copy

grid_manager_main = GridManager()
game_main = Game()

pg.init()

generation = 0

ia_play = True

def main2(genomes, config):
	global generation, grid_manager_main, game_main
	generation += 1

	networks = []
	genomes_list = []
	guns = []
	scores = {}
	grid_manager_copy = copy.deepcopy(grid_manager_main)
	game_copy = copy.deepcopy(game_main)
	
	games = [game_copy] * len(genomes)
	grids = [grid_manager_copy] * len(genomes)
	
	for indice, genome in enumerate(genomes):
		gun = Shooter(pos = BOTTOM_CENTER)
		gun.putInBox()
		
		network = neat.nn.FeedForwardNetwork.create(genome[1], config)
		networks.append(network)
		genome[1].fitness = 0
		genomes_list.append(genome[1])
		guns.append(gun)
		
		background = Background()
		
		grid_manager = grids[indice]

		game = games[indice]

		# Starting mouse position
		mouse_pos = (DISP_W/2, DISP_H/2)
		
		last_move = [0, 0]
		isInitialMove = True
		
		while not game.over:
			for event in pg.event.get():
				if event.type == pg.QUIT:
					pg.quit()
					quit()

				if event.type == pg.KEYDOWN:
					# Ctrl+C to quit
					if event.key == pg.K_SPACE:
						gun.fire()
					if event.key == pg.K_LEFT:
						mouse_pos = (mouse_pos[0] - 10 , mouse_pos[1])
					if event.key == pg.K_RIGHT:
						mouse_pos = (mouse_pos[0] + 10 , mouse_pos[1])
					if event.key == pg.K_c and pg.key.get_mods() & pg.KMOD_CTRL:
						pg.quit()
						quit()
			
			inputs = []
			for j, line in enumerate(grid_manager.grid):
				if j > len(grid_manager.grid)-3 and j != len(grid_manager.grid)-1:
					for bubble in line:
						inputs.append(sum_color(bubble))


			inputs.append(gun.loaded.color[0] + gun.loaded.color[1] + gun.loaded.color[2])
			res = network.activate(inputs)
			mouse_pos = (res[0] * DISP_W, res[1] * DISP_H)

			targetEquals = False

			if not isInitialMove and last_move[0] == mouse_pos[0] and last_move[1] == mouse_pos[1]:
				targetEquals = True
				
	
			last_move[0] = mouse_pos[0]
			last_move[1] = mouse_pos[1]

			
			background.draw()				

			grid_manager.view(gun, game, genomes_list, indice, targetEquals)

			gun.rotate(mouse_pos)			
			gun.fire()

			gun.draw_bullets()					

			
			if game.score > game.prev_score:
				proportion = game.score - game.prev_score
				genomes_list[indice].fitness += 1 * proportion	
			
			game.drawScore()
			
			pg.display.update()
			clock.tick(10000)
			isInitialMove = False
								
		genomes_list[indice].fitness -= 20
		scores[indice] = game.score
		

def sum_color(bubble):
	if bubble.color == BG_COLOR:
		return 0
	
	color_sum = bubble.color[0] + bubble.color[1] + bubble.color[2]

	if color_sum == 256:
		color_sum = 1
	elif color_sum == 257:
		color_sum = 2
	elif color_sum == 258:
		color_sum = 3
	elif color_sum == 420:
		color_sum = 4
	elif color_sum == 510:
		color_sum = 5
	elif color_sum == 382: 
		color_sum = 6

	return color_sum

def run(path_config):
	config = neat.config.Config(neat.DefaultGenome,
								neat.DefaultReproduction,
								neat.DefaultSpeciesSet,
								neat.DefaultStagnation,
								path_config)

	population = neat.Population(config)
	population.add_reporter(neat.StdOutReporter(True))
	population.add_reporter(neat.StatisticsReporter())
	winner = population.run(main2, 3)

if __name__ == '__main__': 
	path = os.path.dirname(__file__)
	path_config = os.path.join(path, 'config.txt')
	run(path_config)

