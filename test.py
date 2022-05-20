
import pygame


def main():

    pygame.init()
    pygame.display.set_caption("minimal program")

    screen = pygame.display.set_mode((960, 960))

    running = True

    # main loop
    while running:
        print("hey")
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False


if __name__ == "__main__":
    main()
