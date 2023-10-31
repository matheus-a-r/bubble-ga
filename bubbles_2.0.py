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

def main2(genomes, config):
	global generation
	generation += 1

	networks = []
	genomes_list = []
	guns = []
	scores = {}


	for indice, genome in enumerate(genomes):
		gun = Shooter(pos = BOTTOM_CENTER)
		gun.putInBox()
		
		network = neat.nn.FeedForwardNetwork.create(genome[1], config)
		networks.append(network)
		genome[1].fitness = 0
		genomes_list.append(genome[1])
		guns.append(gun)

		# MAIN ORIGINAL
		# Create background
		background = Background()

		grid_manager = GridManager()
		game = Game()	

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
			

			old_score = game.score

			inputs = []
			grid = grid_manager.grid
			
			for row in range(GRID_ROWS):
				for col in range(GRID_COLS):
					bubble = grid[row][col]
					inputs.append(row)
					inputs.append(col)
					inputs.append(sum_color(bubble))
					
			inputs.append(gun.loaded.color[0] + gun.loaded.color[1] + gun.loaded.color[2])
			
			res = network.activate(inputs)

			mouse_pos = (res[0] * DISP_W, res[1] * DISP_H)

			if not isInitialMove and last_move[0] == mouse_pos[0] and last_move[1] == mouse_pos[1]:
				genomes_list[indice].fitness -= 1

				
			last_move[0] = mouse_pos[0]
			last_move[1] = mouse_pos[1]
			
			background.draw()				# Draw BG first		

			grid_manager.view(gun, game)	# Check collision with bullet and update grid as needed		

			gun.rotate(mouse_pos)			# Rotate the gun if the mouse is moved	
			gun.fire()	

			gun.draw_bullets()				# Draw and update bullet and reloads	

			new_score = game.score

			game.drawScore()				# draw score

			if new_score > old_score:
				genomes_list[indice].fitness += 5
			else:
				genomes_list[indice].fitness -= 1
			pg.display.update()

			clock.tick(10000)					# 60 FPS
		
		scores[indice] = game.score
		#genomes_list[indice].fitness += game.score/10
	max_score = 0
	index = 0
	for score in scores:
		if scores[score] > max_score:
			max_score = scores[score]
			index = score
	
	guns = [guns[index]]
	networks = [networks[index]]
	genomes_list = [genomes_list[index]]

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
	population.run(main2, 50)

if __name__ == '__main__': 
	path = os.path.dirname(__file__)
	path_config = os.path.join(path, 'config.txt')
	run(path_config)


