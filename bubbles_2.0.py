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

ai_jogando = True
geracao = 0


def main(genomas, config):
	global geracao
	geracao += 1

	if ai_jogando:
		redes = []
		lista_genomas = []
		guns = []
		for _, genoma in genomas:
			rede = neat.nn.FeedForwardNetwork.create(genoma, config)
			redes.append(rede)
			genoma.fitness = 0
			lista_genomas.append(genoma)
			guns.append(Shooter(pos = BOTTOM_CENTER))
	# Create background
	else:
		guns = Shooter(pos = BOTTOM_CENTER)
	background = Background()

	# Initialize gun, position at bottom center of the screen
	
	guns.putInBox()	

	grid_manager = GridManager()
	game = Game()	
	# cheat_manager = CheatManager(grid_manager, gun)

	# Starting mouse position
	mouse_pos = (DISP_W/2, DISP_H/2)
	
	# pretty self-explanatory
	while not game.over:		
		clock.tick(1000)
		# quit when you press the x
		for event in pg.event.get():
			if event.type == pg.QUIT:
				pg.quit()
				quit()

			# get mouse position
			#if event.type == pg.MOUSEMOTION: mouse_pos = pg.mouse.get_pos()
				
			# if you click, fire a bullet
			#if event.type == pg.MOUSEBUTTONDOWN: gun.fire()
			if not ai_jogando:
				if event.type == pg.KEYDOWN:
					#cheat_manager.view(event) # if a key is pressed, the cheat manager should know about it

					# Ctrl+C to quit
					if event.key == pg.K_SPACE:
						guns.fire()
					if event.key == pg.K_LEFT:
						mouse_pos = (mouse_pos[0] - 10 , mouse_pos[1])
					if event.key == pg.K_RIGHT:
						mouse_pos = (mouse_pos[0] + 10 , mouse_pos[1])
					if event.key == pg.K_c and pg.key.get_mods() & pg.KMOD_CTRL:
						pg.quit()
						quit()
		old_score = game.score
		indice_bola = 0
		fired = False
		for i, gun in enumerate(guns):
			gun.rotate(grid_manager.targets[indice_bola])
		# aumentar um pouquinho a fitness do canhao
			output = redes[i].activate((sum(gun.loaded.color)),
										sum(grid_manager.targets[indice_bola]))
		# -1 e 1 -> se o output for > 0.5 então o canhao atira pula
			if output[0] > 0.5:
				gun.fire()
				fired = True
			else:
				indice_bola += 1
			if indice_bola == len(grid_manager.targets):
				gun.fire()
				lista_genomas[i].fitness += 0.1
				fired = True
			if fired:
				if game.score > old_score:
					lista_genomas[i].fitness += 0.1
				else:
					lista_genomas[i].fitness -= 0.1
			
                
		
		background.draw()				# Draw BG first		

		grid_manager.view(guns, game)	# Check collision with bullet and update grid as needed		

		guns.rotate(mouse_pos)			# Rotate the gun if the mouse is moved		
		guns.draw_bullets()				# Draw and update bullet and reloads	

		game.drawScore()				# draw score

		pg.display.update()							# 60 FPS

	game.gameOverScreen(grid_manager, background)

	return

def run(config_path):
    # Carregar a configuração do NEAT a partir do arquivo de configuração
    config = neat.config.Config(neat.DefaultGenome, 
								neat.DefaultReproduction, 
								neat.DefaultSpeciesSet, 
								neat.DefaultStagnation, 
								config_path)

    # Criar a população inicial
    population = neat.Population(config)

    # Adicionar um relatório de estatísticas para acompanhar o progresso
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    # Evoluir a população em várias gerações
    num_generations = 10  # Defina o número de gerações desejado
    for gen in range(num_generations):
        print(f"Geracao {gen + 1}")
        fitness_scores = main(population.run(eval_genomes, 1))
        max_fitness = max(fitness_scores)
        print(f"Melhor pontuação de fitness na geração {gen + 1}: {max_fitness}")

        # Você pode adicionar lógica para salvar os melhores genomas e resultados aqui

    # Após todas as gerações, você pode escolher a melhor rede neural para uso posterior
    best_genome = stats.best_genome()
    best_network = neat.nn.FeedForwardNetwork.create(best_genome, config)

    # Execute o jogo usando a melhor rede neural ou tome outras ações com base nos resultados

if __name__ == '__main__':
    # Substitua 'config.txt' pelo caminho real do seu arquivo de configuração do NEAT
    config_path = 'config.txt'
    run(config_path)


