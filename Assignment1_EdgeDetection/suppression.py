import sobel
import numpy as np

gradient_magnitude = [[]]
gradient_direction = [[]]
row = 0
col = 0


def find_edge(y, x, y1, x1, y2, x2):
    if y1 < 0 or y1 >= row or x1 < 0 or x1 >= col:
        if y2 < 0 or y2 >= row or x2 < 0 or x2 >= col:
            return gradient_magnitude[y][x]
        if gradient_magnitude[y][x] < gradient_magnitude[y2][x2]:
            return 0
        else:
            return gradient_magnitude[y][x]
    if y2 < 0 or y2 >= row or x2 < 0 or x2 >= col:
        if gradient_magnitude[y][x] < gradient_magnitude[y1][x1]:
            return 0
        else:
            return gradient_magnitude[y][x]
    if gradient_magnitude[y][x] < gradient_magnitude[y1][x1] or gradient_magnitude[y][x] < gradient_magnitude[y2][x2]:
        return 0
    else:
        return gradient_magnitude[y][x]


def find_orientation(y, x, theta):
    if theta >= 67.5 or theta < -67.5:
        #TODO: left-right
        return find_edge(y, x, y-1, x, y+1, x)
    elif 22.5 <= theta < 67.5:
        #TODO: topleft-botright
        return find_edge(y, x, y - 1, x + 1, y - 1, x + 1)
    elif -22.5 <= theta < 22.5:
        #TODO: top-bot
        return find_edge(y, x, y, x-1,  y, x+1)
    elif -67.6 <= theta < -22.5:
        #TODO: topright-botleft
        return find_edge(y, x, y + 1, x + 1, y + 1, x + 1)


def suppression(source, threshold=0):
    global row, col, gradient_direction, gradient_magnitude
    gradient_magnitude, gradient_direction = sobel.sobel_filter(source, threshold)
    row, col = gradient_magnitude.shape
    suppressed_matrix = np.zeros((row, col))
    for i in range(row):
        for j in range(col):
            suppressed_matrix[i][j] = find_orientation(i, j, gradient_direction[i][j])

    return suppressed_matrix

