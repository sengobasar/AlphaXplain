import pygame
import sys


class GridRenderer:

    def __init__(self, size=5, cell=100):

        pygame.init()

        self.size = size
        self.cell = cell

        self.width = size * cell
        self.height = size * cell

        self.screen = pygame.display.set_mode(
            (self.width, self.height)
        )

        pygame.display.set_caption("RL GridWorld")

        self.clock = pygame.time.Clock()

        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.BLUE = (50, 100, 255)   # Agent
        self.GREEN = (50, 200, 50)   # Goal

    def draw_grid(self):

        for i in range(self.size + 1):

            pygame.draw.line(
                self.screen,
                self.BLACK,
                (0, i * self.cell),
                (self.width, i * self.cell),
                2
            )

            pygame.draw.line(
                self.screen,
                self.BLACK,
                (i * self.cell, 0),
                (i * self.cell, self.height),
                2
            )

    def render(self, agent_pos, goal_pos):

        # Close window safely
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        self.screen.fill(self.WHITE)

        self.draw_grid()

        # Draw goal
        gx, gy = goal_pos
        pygame.draw.rect(
            self.screen,
            self.GREEN,
            (
                gy * self.cell,
                gx * self.cell,
                self.cell,
                self.cell
            )
        )

        # Draw agent
        ax, ay = agent_pos
        pygame.draw.rect(
            self.screen,
            self.BLUE,
            (
                ay * self.cell,
                ax * self.cell,
                self.cell,
                self.cell
            )
        )

        pygame.display.flip()

        self.clock.tick(10)  # FPS (speed)
