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
	global generation
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
	
	game = Game()	

	# Starting mouse position
	mouse_pos = (DISP_W/2, DISP_H/2)
	
	# pretty self-explanatory
	while not game.over:		

		# quit when you press the x
		if not ia_play:
			for event in pg.event.get():
				if event.type == pg.QUIT:
					pg.quit()
					quit()

				# get mouse position
				#if event.type == pg.MOUSEMOTION: mouse_pos = pg.mouse.get_pos()
					
				# if you click, fire a bullet
				#if event.type == pg.MOUSEBUTTONDOWN: gun.fire()
				
				if event.type == pg.KEYDOWN:
					cheat_manager.view(event) # if a key is pressed, the cheat manager should know about it

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
		else:
			for gun in guns:

				background.draw()				# Draw BG first		

				grid_manager.view(gun, game)	# Check collision with bullet and update grid as needed		

				gun.rotate(mouse_pos)			# Rotate the gun if the mouse is moved		
				gun.draw_bullets()				# Draw and update bullet and reloads	

				game.drawScore()				# draw score

				pg.display.update()		
				clock.tick(60)					# 60 FPS

			game.gameOverScreen(grid_manager, background)

	return

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


