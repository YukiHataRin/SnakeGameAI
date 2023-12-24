import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
pygame.init()
font = pygame.font.Font('ttf/arial.ttf', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)
HEAD_COLOR = (0, 229, 238)    
BODY_COLOR = (255, 255, 0)
TAIL_COLOR = WHITE

BLOCK_SIZE = 20

class SnakeGameAI:
    def __init__(self, w=240, h=240, speed=100):
        self.w = w
        self.h = h
        self.speed = speed
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [self.head, Point(self.head.x - BLOCK_SIZE, self.head.y), 
                      Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action):
        self.frame_iteration += 1

        for event in pygame.event.get():
            if event.type is pygame.QUIT:
                pygame.quit()
                quit()

        # 2. move
        self._move(action)
        self.snake.insert(0, self.head)

        # 3. check if game over
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            # 減少的分數與蛇的長度成反比
            penalty = -20 + len(self.snake) * 0.05
            reward = max(penalty, -20)  # 確保最低分不低於-10
            reward = min(0, -10)
            return reward, game_over, self.score

        # Calculate distance to food before and after moving
        distance_before_move = self._distance_to_food(self.head)
        distance_after_move = self._distance_to_food(self.snake[0])

        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            # 增加的分數與蛇的長度成正比
            bonus = 10 + len(self.snake) * 0.05
            reward = bonus
            self._place_food()
        else:
            self.snake.pop()
            # Additional rewards or penalties based on distance to food
            if distance_after_move < distance_before_move:
                reward += (1 / len(self.snake))
            else:
                reward -= (1 / len(self.snake))

        self._update_ui()
        self.clock.tick(self.speed)
        return reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        if pt in self.snake[1:]:
            return True

        return False

    def _update_ui(self):
        self.display.fill(BLACK)

        # 獲取蛇的頭部和尾部位置
        head = self.snake[0]
        tail = self.snake[-1]

        # 繪製蛇的每一個部分
        for idx, pt in enumerate(self.snake):
            if pt == head:
                # 繪製蛇頭
                pygame.draw.rect(self.display, HEAD_COLOR, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            elif pt == tail:
                # 繪製蛇尾
                pygame.draw.rect(self.display, TAIL_COLOR, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            else:
                # 繪製蛇的身體
                pygame.draw.rect(self.display, BODY_COLOR, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))

        # 繪製食物
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        # 顯示得分
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]  # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]  # right turn r -> d -> l -> u
        else:  # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]  # left turn r -> u -> l -> d

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)

    def _distance_to_food(self, point):
        return abs(point.x - self.food.x) + abs(point.y - self.food.y)

# if __name__ == "__main__":
#     game = SnakeGameAI()
#     while True:
#         game.play_step([1, 0, 0])  # replace with your logic or AI input
