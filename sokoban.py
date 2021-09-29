import sys
from typing import Deque
import pygame
import queue
import collections
import numpy as np
import heapq
import time
import os

# Game code begins here


class game:

    def is_valid_value(self, char):
        if (char == ' ' or  # floor
            char == '#' or  # wall
            char == '@' or  # worker on floor
            char == '.' or  # dock
            char == '*' or  # box on dock
            char == '$' or  # box
            char == '+'):   # worker on dock
            return True
        else:
            return False

    def __init__(self, level):
        self.queue = queue.LifoQueue()
        self.matrix = []

        file = open(os.path.dirname(os.path.abspath(__file__)) +
                    '/levels' + '/level' + str(level), 'r')

        for line in file:
            row = []
            if line.strip() != "":
                row = []
                for c in line:
                    if c != '\n' and self.is_valid_value(c):
                        row.append(c)
                    elif c == '\n':  # jump to next row when newline
                        continue
                    else:
                        print(("ERROR: Level ") + str(level) +
                              (" has invalid value ") + c)
                        sys.exit(1)
                self.matrix.append(row)
            else:
                break

    def load_size(self):  # return size of game
        x = 0
        y = len(self.matrix)
        for row in self.matrix:
            if len(row) > x:
                x = len(row)
        return (x * 32, y * 32)

    def get_matrix(self):  # return matrix of game
        return self.matrix

    def print_matrix(self):  # print game's matrix
        for row in self.matrix:
            for char in row:
                sys.stdout.write(char)
                sys.stdout.flush()
            sys.stdout.write('\n')

    def get_content(self, x, y):
        return self.matrix[y][x]

    def set_content(self, x, y, content):
        if self.is_valid_value(content):
            self.matrix[y][x] = content
        else:
            print(("ERROR: Value '") + content +
                  ("' to be added is not valid"))

    def worker(self):
        x = 0
        y = 0
        for row in self.matrix:
            for pos in row:
                if pos == '@' or pos == '+':
                    return (x, y, pos)
                else:
                    x = x + 1
            y = y + 1
            x = 0

    def can_move(self, x, y):
        return self.get_content(self.worker()[0]+x, self.worker()[1]+y) not in ['#', '*', '$']

    def next(self, x, y):
        return self.get_content(self.worker()[0]+x, self.worker()[1]+y)

    def can_push(self, x, y):
        return (self.next(x, y) in ['*', '$'] and self.next(x+x, y+y) in [' ', '.'])

    def is_completed(self):
        for row in self.matrix:
            for cell in row:
                if cell == '$':
                    return False
        return True

    def move_box(self, x, y, a, b):
        #        (x,y) -> move to do
        #        (a,b) -> box to move
        current_box = self.get_content(x, y)
        future_box = self.get_content(x+a, y+b)
        if current_box == '$' and future_box == ' ':
            self.set_content(x+a, y+b, '$')
            self.set_content(x, y, ' ')
        elif current_box == '$' and future_box == '.':
            self.set_content(x+a, y+b, '*')
            self.set_content(x, y, ' ')
        elif current_box == '*' and future_box == ' ':
            self.set_content(x+a, y+b, '$')
            self.set_content(x, y, '.')
        elif current_box == '*' and future_box == '.':
            self.set_content(x+a, y+b, '*')
            self.set_content(x, y, '.')

    def unmove(self):
        if not self.queue.empty():
            movement = self.queue.get()
            if movement[2]:
                current = self.worker()
                self.move(movement[0] * -1, movement[1] * -1, False)
                self.move_box(current[0]+movement[0], current[1] +
                              movement[1], movement[0] * -1, movement[1] * -1)
            else:
                self.move(movement[0] * -1, movement[1] * -1, False)

    def move(self, x, y, save):
        if self.can_move(x, y):
            current = self.worker()
            future = self.next(x, y)
            if current[2] == '@' and future == ' ':
                self.set_content(current[0]+x, current[1]+y, '@')
                self.set_content(current[0], current[1], ' ')
                if save:
                    self.queue.put((x, y, False))
            elif current[2] == '@' and future == '.':
                self.set_content(current[0]+x, current[1]+y, '+')
                self.set_content(current[0], current[1], ' ')
                if save:
                    self.queue.put((x, y, False))
            elif current[2] == '+' and future == ' ':
                self.set_content(current[0]+x, current[1]+y, '@')
                self.set_content(current[0], current[1], '.')
                if save:
                    self.queue.put((x, y, False))
            elif current[2] == '+' and future == '.':
                self.set_content(current[0]+x, current[1]+y, '+')
                self.set_content(current[0], current[1], '.')
                if save:
                    self.queue.put((x, y, False))
        elif self.can_push(x, y):
            current = self.worker()
            future = self.next(x, y)
            future_box = self.next(x+x, y+y)
            if current[2] == '@' and future == '$' and future_box == ' ':
                self.move_box(current[0]+x, current[1]+y, x, y)
                self.set_content(current[0], current[1], ' ')
                self.set_content(current[0]+x, current[1]+y, '@')
                if save:
                    self.queue.put((x, y, True))
            elif current[2] == '@' and future == '$' and future_box == '.':
                self.move_box(current[0]+x, current[1]+y, x, y)
                self.set_content(current[0], current[1], ' ')
                self.set_content(current[0]+x, current[1]+y, '@')
                if save:
                    self.queue.put((x, y, True))
            elif current[2] == '@' and future == '*' and future_box == ' ':
                self.move_box(current[0]+x, current[1]+y, x, y)
                self.set_content(current[0], current[1], ' ')
                self.set_content(current[0]+x, current[1]+y, '+')
                if save:
                    self.queue.put((x, y, True))
            elif current[2] == '@' and future == '*' and future_box == '.':
                self.move_box(current[0]+x, current[1]+y, x, y)
                self.set_content(current[0], current[1], ' ')
                self.set_content(current[0]+x, current[1]+y, '+')
                if save:
                    self.queue.put((x, y, True))
            if current[2] == '+' and future == '$' and future_box == ' ':
                self.move_box(current[0]+x, current[1]+y, x, y)
                self.set_content(current[0], current[1], '.')
                self.set_content(current[0]+x, current[1]+y, '@')
                if save:
                    self.queue.put((x, y, True))
            elif current[2] == '+' and future == '$' and future_box == '.':
                self.move_box(current[0]+x, current[1]+y, x, y)
                self.set_content(current[0], current[1], '.')
                self.set_content(current[0]+x, current[1]+y, '@')
                if save:
                    self.queue.put((x, y, True))
            elif current[2] == '+' and future == '*' and future_box == ' ':
                self.move_box(current[0]+x, current[1]+y, x, y)
                self.set_content(current[0], current[1], '.')
                self.set_content(current[0]+x, current[1]+y, '+')
                if save:
                    self.queue.put((x, y, True))
            elif current[2] == '+' and future == '*' and future_box == '.':
                self.move_box(current[0]+x, current[1]+y, x, y)
                self.set_content(current[0], current[1], '.')
                self.set_content(current[0]+x, current[1]+y, '+')
                if save:
                    self.queue.put((x, y, True))


def print_game(matrix, screen):
    screen.fill(background)
    x = 0
    y = 0
    for row in matrix:
        for char in row:
            if char == ' ':  # floor
                screen.blit(floor, (x, y))
            elif char == '#':  # wall
                screen.blit(wall, (x, y))
            elif char == '@':  # worker on floor
                screen.blit(worker, (x, y))
            elif char == '.':  # dock
                screen.blit(docker, (x, y))
            elif char == '*':  # box on dock
                screen.blit(box_docked, (x, y))
            elif char == '$':  # box
                screen.blit(box, (x, y))
            elif char == '+':  # worker on dock
                screen.blit(worker_docked, (x, y))
            x = x + 32
        x = 0
        y = y + 32


def get_key():
    while 1:
        event = pygame.event.poll()
        if event.type == pygame.KEYDOWN:
            return event.key
        else:
            pass


def display_box(screen, message):
    "Print a message in a box in the middle of the screen"
    fontobject = pygame.font.Font(None, 18)
    pygame.draw.rect(screen, (0, 0, 0),
                     ((screen.get_width() / 2) - 100,
                      (screen.get_height() / 2) - 10,
                      200, 20), 0)
    pygame.draw.rect(screen, (255, 255, 255),
                     ((screen.get_width() / 2) - 102,
                      (screen.get_height() / 2) - 12,
                      204, 24), 1)
    if len(message) != 0:
        screen.blit(fontobject.render(message, 1, (255, 255, 255)),
                    ((screen.get_width() / 2) - 100, (screen.get_height() / 2) - 10))
    pygame.display.flip()


def display_end(screen):
    message = "Level Completed"
    fontobject = pygame.font.Font(None, 18)
    pygame.draw.rect(screen, (0, 0, 0),
                     ((screen.get_width() / 2) - 100,
                      (screen.get_height() / 2) - 10,
                      200, 20), 0)
    pygame.draw.rect(screen, (255, 255, 255),
                     ((screen.get_width() / 2) - 102,
                      (screen.get_height() / 2) - 12,
                      204, 24), 1)
    screen.blit(fontobject.render(message, 1, (255, 255, 255)),
                ((screen.get_width() / 2) - 100, (screen.get_height() / 2) - 10))
    pygame.display.flip()

# Game code ends here

# Algorithm code begins here----------------------------------------------------------------


class PriorityQueue:
    # Define a PriorityQueue data structure that will be used
    def __init__(self):
        self.Heap = []
        self.Count = 0

    def push(self, item, priority):
        entry = (priority, self.Count, item)
        heapq.heappush(self.Heap, entry)
        self.Count += 1

    def pop(self):
        (_, _, item) = heapq.heappop(self.Heap)
        return item

    def isEmpty(self):
        return len(self.Heap) == 0

# Load puzzles and define the rules of sokoban


def transferToGameState(layout):
    # Transfer the layout of initial puzzle
    layout = [x.replace('\n', '') for x in layout]
    layout = [','.join(layout[i]) for i in range(len(layout))]
    layout = [x.split(',') for x in layout]
    maxColsNum = max([len(x) for x in layout])
    for irow in range(len(layout)):
        for icol in range(len(layout[irow])):
            if layout[irow][icol] == ' ':
                layout[irow][icol] = 0   # floor
            elif layout[irow][icol] == '#':
                layout[irow][icol] = 1  # wall
            elif layout[irow][icol] == '@':
                layout[irow][icol] = 2  # worker
            elif layout[irow][icol] == '$':
                layout[irow][icol] = 3  # box
            elif layout[irow][icol] == '.':
                layout[irow][icol] = 4  # dock
            elif layout[irow][icol] == '+':
                layout[irow][icol] = 5  # work on dock
            elif layout[irow][icol] == '*':
                layout[irow][icol] = 6  # box on dock
        colsNum = len(layout[irow])
        if colsNum < maxColsNum:
            layout[irow].extend([1 for _ in range(maxColsNum-colsNum)])
    return np.array(layout)


def PosOfPlayer(gameState):
    # Return the position of agent
    # e.g. (2, 2)
    return tuple(np.argwhere(gameState == 2 | (gameState == 5))[0])


def PosOfBoxes(gameState):
    # Return the positions of boxes
    # e.g. ((2, 3), (3, 4), (4, 4), (6, 1), (6, 4), (6, 5))
    return tuple(tuple(x) for x in np.argwhere((gameState == 3) | (gameState == 6)))


def PosOfWalls(gameState):
    # Return the positions of walls
    # e.g. like those above
    return tuple(tuple(x) for x in np.argwhere(gameState == 1))


def PosOfGoals(gameState):
    # Return the positions of goals
    # e.g. like those above
    return tuple(tuple(x) for x in np.argwhere((gameState == 4) | (gameState == 6)))


def isEndState(posBox):
    # Check if all boxes are on the goals (i.e. pass the game)
    return sorted(posBox) == sorted(posGoals)


def isLegalAction(action, posPlayer, posBox):
    # Check if the given action is legal
    xPlayer, yPlayer = posPlayer
    if action[-1].isupper():  # the move was a push
        x1, y1 = xPlayer + 2 * action[0], yPlayer + 2 * action[1]
    else:
        x1, y1 = xPlayer + action[0], yPlayer + action[1]
    return (x1, y1) not in posBox + posWalls


def legalActions(posPlayer, posBox):
    # Return all legal actions for the agent in the current game state
    allActions = [[-1, 0, 'u', 'U'], [1, 0, 'd', 'D'],
                  [0, -1, 'l', 'L'], [0, 1, 'r', 'R']]
    xPlayer, yPlayer = posPlayer
    legalActions = []
    for action in allActions:
        x1, y1 = xPlayer + action[0], yPlayer + action[1]
        if (x1, y1) in posBox:  # the move was a push
            action.pop(2)  # drop the little letter
        else:
            action.pop(3)  # drop the upper letter
        if isLegalAction(action, posPlayer, posBox):
            legalActions.append(action)
        else:
            continue
    # e.g. ((0, -1, 'l'), (0, 1, 'R'))
    return tuple(tuple(x) for x in legalActions)


def updateState(posPlayer, posBox, action):
    # Return updated game state after an action is taken
    xPlayer, yPlayer = posPlayer  # the previous position of player
    newPosPlayer = [xPlayer + action[0], yPlayer +
                    action[1]]  # the current position of player
    posBox = [list(x) for x in posBox]
    if action[-1].isupper():  # if pushing, update the position of box
        posBox.remove(newPosPlayer)
        posBox.append([xPlayer + 2 * action[0], yPlayer + 2 * action[1]])
    posBox = tuple(tuple(x) for x in posBox)
    newPosPlayer = tuple(newPosPlayer)
    return newPosPlayer, posBox


def isFailed(posBox):
    # This function used to observe if the state is potentially failed, then prune the search
    rotatePattern = [[0, 1, 2, 3, 4, 5, 6, 7, 8],
                     [2, 5, 8, 1, 4, 7, 0, 3, 6],
                     [0, 1, 2, 3, 4, 5, 6, 7, 8][::-1],
                     [2, 5, 8, 1, 4, 7, 0, 3, 6][::-1]]
    flipPattern = [[2, 1, 0, 5, 4, 3, 8, 7, 6],
                   [0, 3, 6, 1, 4, 7, 2, 5, 8],
                   [2, 1, 0, 5, 4, 3, 8, 7, 6][::-1],
                   [0, 3, 6, 1, 4, 7, 2, 5, 8][::-1]]
    allPattern = rotatePattern + flipPattern

    for box in posBox:
        if box not in posGoals:
            board = [(box[0] - 1, box[1] - 1), (box[0] - 1, box[1]), (box[0] - 1, box[1] + 1),
                     (box[0], box[1] - 1), (box[0],
                                            box[1]), (box[0], box[1] + 1),
                     (box[0] + 1, box[1] - 1), (box[0] + 1, box[1]), (box[0] + 1, box[1] + 1)]
            for pattern in allPattern:
                newBoard = [board[i] for i in pattern]
                if newBoard[1] in posWalls and newBoard[5] in posWalls:
                    return True
                elif newBoard[1] in posBox and newBoard[2] in posWalls and newBoard[5] in posWalls:
                    return True
                elif newBoard[1] in posBox and newBoard[2] in posWalls and newBoard[5] in posBox:
                    return True
                elif newBoard[1] in posBox and newBoard[2] in posBox and newBoard[5] in posBox:
                    return True
                elif newBoard[1] in posBox and newBoard[6] in posBox and newBoard[2] in posWalls and newBoard[3] in posWalls and newBoard[8] in posWalls:
                    return True
    return False

# Implement all approcahes


def breadthFirstSearch():
    # Implement breadthFirstSearch approach
    start = time.time()
    beginBox = PosOfBoxes(gameState)
    beginPlayer = PosOfPlayer(gameState)

    # e.g. ((2, 2), ((2, 3), (3, 4), (4, 4), (6, 1), (6, 4), (6, 5)))
    startState = (beginPlayer, beginBox)
    frontier = collections.deque([[startState]])  # store states
    actions = collections.deque([[0]])  # store actions
    exploredSet = set()
    res = []
    while frontier:
        node = frontier.popleft()
        node_action = actions.popleft()
        if isEndState(node[-1][-1]):
            res.append(','.join(node_action[1:]).replace(',', ''))
            break
        if node[-1] not in exploredSet:
            exploredSet.add(node[-1])
            for action in legalActions(node[-1][0], node[-1][1]):
                newPosPlayer, newPosBox = updateState(
                    node[-1][0], node[-1][1], action)
                if isFailed(newPosBox):
                    continue
                frontier.append(node + [(newPosPlayer, newPosBox)])
                actions.append(node_action + [action[-1]])
    end = time.time()
    print('RUNTIME OF BREATHFIRST SEARCH: %.3f second.' % (end - start),
          'EXPLORED NODE                : %d' % (len(exploredSet)),
          'NUMBER OF STEP               : %d' % (len(res[0])), sep='\n')
    return res


def heuristic(posPlayer, posBox):
    # A heuristic function to calculate the overall distance between the else boxes and the else goals
    distance = 0
    completes = set(posGoals) & set(posBox)
    sortposBox = list(set(posBox).difference(completes))
    sortposGoals = list(set(posGoals).difference(completes))
    for i in range(len(sortposBox)):
        distance += (abs(sortposBox[i][0] - sortposGoals[i][0])) + \
            (abs(sortposBox[i][1] - sortposGoals[i][1]))
    return distance


def cost(actions):
    # A cost function
    return len([x for x in actions if x.islower()])


def aStarSearch():
    # Implement aStarSearch approach
    start = time.time()
    beginBox = PosOfBoxes(gameState)
    beginPlayer = PosOfPlayer(gameState)

    start_state = (beginPlayer, beginBox)
    frontier = PriorityQueue()
    frontier.push([start_state], heuristic(beginPlayer, beginBox))
    exploredSet = set()
    actions = PriorityQueue()
    actions.push([0], heuristic(beginPlayer, start_state[1]))
    res = []
    while frontier:
        node = frontier.pop()
        node_action = actions.pop()
        if isEndState(node[-1][-1]):
            res.append(','.join(node_action[1:]).replace(',', ''))
            break
        if node[-1] not in exploredSet:
            exploredSet.add(node[-1])
            Cost = cost(node_action[1:])
            for action in legalActions(node[-1][0], node[-1][1]):
                newPosPlayer, newPosBox = updateState(
                    node[-1][0], node[-1][1], action)
                if isFailed(newPosBox):
                    continue
                Heuristic = heuristic(newPosPlayer, newPosBox)
                frontier.push(
                    node + [(newPosPlayer, newPosBox)], Heuristic + Cost)
                actions.push(node_action + [action[-1]], Heuristic + Cost)
    end = time.time()
    print('RUNTIME OF A* SEARCH: %.3f second.' % (end - start),
          'EXPLORED NODE       : %d' % (len(exploredSet)),
          'NUMBER OF STEP      : %d' % (len(res[0])), sep='\n')

    return res

# Read command


def readCommand(argv):
    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option('-l', '--level', dest='sokobanLevels', type='int',
                      help='level of game to play', default=1)
    parser.add_option('-m', '--method', dest='agentMethod', type='string',
                      help='research method', default='bfs')
    args = dict()

    options, _ = parser.parse_args(argv)

    if int(options.sokobanLevels) < 1 or int(options.sokobanLevels) > 20:
        print(("ERROR: Level ") + str(options.sokobanLevels) + (" is out of range"))
        sys.exit(1)
    else:
        with open(os.path.dirname(os.path.abspath(__file__)) +
                  '/levels' + '/level' + str(options.sokobanLevels), 'r') as f:
            layout = f.readlines()

    args['layout'] = layout
    args['method'] = options.agentMethod
    args['level'] = options.sokobanLevels
    return args


# Algorithm code ends here-----------------------------------------------------------------------
pygame.display.set_caption(("AI Project 1: Sokoban"))
icon = pygame.image.load('images/bk-icon.png')
pygame.display.set_icon(icon)
wall = pygame.image.load('images/wall2.png')
floor = pygame.image.load('images/floor3.png')
box = pygame.image.load('images/box3.png')
box_docked = pygame.image.load('images/box_docked3.png')
worker = pygame.image.load('images/worker3.png')
worker_docked = pygame.image.load('images/worker_dock3.png')
docker = pygame.image.load('images/dock3.png')
background = 255, 226, 191
pygame.init()

if __name__ == '__main__':
    m = []
    layout, method, level = readCommand(sys.argv[1:]).values()
    gameState = transferToGameState(layout)
    posWalls = PosOfWalls(gameState)
    posGoals = PosOfGoals(gameState)
    if method == 'astar':
        m = aStarSearch()
    elif method == 'bfs':
        m = breadthFirstSearch()
    else:
        raise ValueError('Invalid method.')
    a = m[0]
    game = game(level)
    size = game.load_size()
    screen = pygame.display.set_mode(size)
    while 1:
        if game.is_completed():
            display_end(screen)
        print_game(game.get_matrix(), screen)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    for i in range(len(a)):
                        if a[0] == 'u' or a[0] == 'U':
                            game.move(0, -1, True)
                            a = a[1:]

                        elif a[0] == 'd' or a[0] == 'D':
                            game.move(0, 1, True)
                            a = a[1:]

                        elif a[0] == 'r' or a[0] == 'R':
                            game.move(1, 0, True)
                            a = a[1:]

                        elif a[0] == 'l' or a[0] == 'L':
                            game.move(-1, 0, True)
                            a = a[1:]

                        else:
                            continue
                        pygame.display.update()

                elif event.key == pygame.K_q:
                    sys.exit(0)
                elif event.key == pygame.K_d:
                    game.unmove()
        pygame.display.update()
