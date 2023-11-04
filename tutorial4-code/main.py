import pygame
import time
import math
import neat
import os
from utils import scale_image, blit_rotate_center, blit_text_center
pygame.font.init()

GRASS = scale_image(pygame.image.load("imgs/grass.jpg"), 2.5)
TRACK = scale_image(pygame.image.load("imgs/track.png"), 0.9)

TRACK_BORDER = scale_image(pygame.image.load("imgs/track-border.png"), 0.9)
TRACK_BORDER_MASK = pygame.mask.from_surface(TRACK_BORDER)
FINISH = pygame.image.load("imgs/finish.png")
FINISH_MASK = pygame.mask.from_surface(FINISH)
FINISH_POSITION = (130, 250)

RED_CAR = scale_image(pygame.image.load("imgs/red-car.png"), 0.55)
GREEN_CAR = scale_image(pygame.image.load("imgs/green-car.png"), 0.55)

WIDTH, HEIGHT = TRACK.get_width(), TRACK.get_height()
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Racing Game!")

MAIN_FONT = pygame.font.SysFont("comicsans", 44)

FPS = 60
PATH = [(175, 119), (110, 70), (56, 133), (70, 481), (318, 731), (404, 680), (418, 521), (507, 475), (600, 551), (613, 715), (736, 713),
        (734, 399), (611, 357), (409, 343), (433, 257), (697, 258), (738, 123), (581, 71), (303, 78), (275, 377), (176, 388), (178, 260)]


class GameInfo:

    def __init__(self):
        self.started = False

    def reset(self):
        self.started = False

    def start(self):
        self.started = True

class AbstractCar:
    def __init__(self, max_vel, rotation_vel):
        self.img = self.IMG
        self.max_vel = max_vel
        self.vel = 0
        self.rotation_vel = rotation_vel
        self.angle = 0
        self.x, self.y = self.START_POS
        self.acceleration = 0.1

    def rotate(self, left=False, right=False):
        if left:
            self.angle += self.rotation_vel
        elif right:
            self.angle -= self.rotation_vel

    def draw(self, win):
        blit_rotate_center(win, self.img, (self.x, self.y), self.angle)

    def move_forward(self):
        self.vel = min(self.vel + self.acceleration, self.max_vel)
        self.move()

    def move_backward(self):
        self.vel = max(self.vel - self.acceleration, -self.max_vel/2)
        self.move()

    def move(self):
        radians = math.radians(self.angle)
        vertical = math.cos(radians) * self.vel
        horizontal = math.sin(radians) * self.vel

        self.y -= vertical
        self.x -= horizontal

    def collide(self, mask, x=0, y=0):
        car_mask = pygame.mask.from_surface(self.img)
        offset = (int(self.x - x), int(self.y - y))
        poi = mask.overlap(car_mask, offset)
        return poi
    
    def cal_distance(self, TRACK_BORDER_MASK, x=0, y=0):
        offset = (int(self.x - x), int(self.y - y))
        return offset
        
    def reset(self):
        self.x, self.y = self.START_POS
        self.angle = 0
        self.vel = 0


class PlayerCar(AbstractCar):
    IMG = RED_CAR
    START_POS = (180, 200)

    def reduce_speed(self):
        self.vel = max(self.vel - self.acceleration / 2, 0)
        self.move()

    def bounce(self):
        self.vel = -self.vel
        self.move()  


def draw(win, images, cars):
    for img, pos in images:
        win.blit(img, pos)

    for car in cars:
        car.draw(win)
        pygame.display.update()


def move_player(player_car, output):
    
    player_car.rotate(left=True)
    player_car.rotate(right=True)
    player_car.move_forward()
    player_car.move_backward()
    
    # keys = pygame.key.get_pressed()
    # moved = False

    # if keys[pygame.K_a]:
    #     player_car.rotate(left=True)
    # if keys[pygame.K_d]:
    #     player_car.rotate(right=True)
    # if keys[pygame.K_w]:
    #     moved = True
    #     player_car.move_forward()
    # if keys[pygame.K_s]:
    #     moved = True
    #     player_car.move_backward()

    # if not moved:
    #     player_car.reduce_speed()


def handle_collision(player_car, game_info):
    if player_car.collide(TRACK_BORDER_MASK) != None:
        player_car.bounce()

    player_finish_poi_collide = player_car.collide(
        FINISH_MASK, *FINISH_POSITION)
    if player_finish_poi_collide != None:
        if player_finish_poi_collide[1] == 0:
            player_car.bounce()
        else:
            player_car.reset()

def main(genomes, config):

    clock = pygame.time.Clock()
    images = [(GRASS, (0, 0)), (TRACK, (0, 0)),
            (FINISH, FINISH_POSITION), (TRACK_BORDER, (0, 0))]
    game_info = GameInfo()

    cars = []
    nets =[]
    ge_list = []

    for _, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        genome.fitness = 0
        ge_list.append(genome)
        cars.append(PlayerCar(4, 4))
    car = PlayerCar(4, 4)
    run = True
    while run:
        clock.tick(FPS)
        
        draw(WIN, images, cars)

        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()

        for i, car in enumerate(cars):

            bx, by = car.cal_distance(TRACK_BORDER_MASK)
            
            inputs = [car.vel, car.x, car.y, bx, by]

            output = nets[i].activate(inputs)
            
            move_player(car, output)

            ge_list[i].fitness += 0.1

        handle_collision(car, game_info)
        

def run(path_config):
	config = neat.config.Config(neat.DefaultGenome,
								neat.DefaultReproduction,
								neat.DefaultSpeciesSet,
								neat.DefaultStagnation,
								path_config)

	population = neat.Population(config)
	population.add_reporter(neat.StdOutReporter(True))
	population.add_reporter(neat.StatisticsReporter())
	winner = population.run(main, 50)

if __name__ == '__main__': 
	path = os.path.dirname(__file__)
	path_config = os.path.join(path, 'config.txt')
	run(path_config)


pygame.quit()
