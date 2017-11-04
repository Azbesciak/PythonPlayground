import math as m

import matplotlib.pyplot as plt
import numpy as np

from mine import common as gr

POSITION_BORDER_LIMIT = 0.5

SUN_HEIGHT = 50

antialiasing_multiplier = 4


def hsv_to_rgb(hsv):
    return gr.hsv2rgb(hsv[0], hsv[1], hsv[2])


colors_space = ((gr.hsv_angle(120), 1, 1), (gr.hsv_angle(0), 1, 1))


def get_h_value(value, max_min_dif, minimum):
    return gr.linear_interpolation(colors_space, (value - minimum) / max_min_dif)[0]


# Funkcja rysująca mapę
def drawMap(mapa, file_name):
    fig, axs = plt.subplots(1, 1, figsize=(8, 6), dpi=80)
    im = axs.imshow(mapa, interpolation="hermite")
    im.set_extent([0, 500, 500, 0])
    plt.show()
    fig.savefig(file_name)


# Wczytywanie punktów na mapie oraz wyskości, szerokości mapy i dystansu między punktami
def loadMapPoints(file_name):
    with open(file_name) as file:
        readline = file.readline()
        height, width, distance = (int(i) for i in readline.split(" "))
        matrix = []
        for i in range(int(height)):
            line = file.readline().replace("\n", "").split(" ")
            vals = [float(i) for i in line]
            matrix.append(vals)
    return matrix, width, height, distance


# Tworzenie macierzy kolorów HSV
def createHSVmatrix(width, height):
    hsvMatrix = []
    for x in range(width):
        hsvMatrix.append([])
        for y in range(height):
            hsvMatrix[x].append([0, 1, 1])
    return hsvMatrix


def simple_shading(mapa, width, height):
    minimum = np.min(mapa)
    max_min_dif = np.max(mapa) - minimum
    mapaHSV = createHSVmatrix(width, height)
    for x in range(width):
        for y in range(height):
            mapaHSV[x][y][0] = get_h_value(mapa[x][y], max_min_dif, minimum)
            if y == 0:
                div = mapa[x][y] - mapa[x][y+1] # Różnica między wysokością punktu a jego prawym sąsiadem
            else:
                div = mapa[x][y] - mapa[x][y-1] # Różnica między wysokością punktu a jego lewym sąsiadem
            div = div*7 / max_min_dif
            if div > 0:
                mapaHSV[x][y][1] -= abs(div)
            else:
                mapaHSV[x][y][2] -= abs(div)
            mapaHSV[x][y] = hsv_to_rgb(mapaHSV[x][y])
    return mapaHSV


# Określanie koloru i cieniowania na podstawie kąta pomiędzy wektorem normalnym powierzchni a wektorem słońca
def vectorShading(mapa, width, height, distance):
    minimum = np.min(mapa)
    height_dif = np.max(mapa) - minimum
    sun = np.array([-distance, SUN_HEIGHT, -distance])  # Wektor słońca
    mapaHSV = createHSVmatrix(width, height)  # Macierz, która jest uzupełniana kolorami HSV na podstawie obliczeń
    matrixOfAngles = prepare_angles_map(distance, width, height, mapa, sun)# Posortowana lista kątów. Żeby lepiej uwidocznić cieniowanie
    reshade_map(width, height, mapa, mapaHSV, matrixOfAngles, height_dif, minimum)
    # return antialiase(mapaHSV)
    return mapaHSV


def prepare_angles_map(distance, width, height, mapa, sun):
    angles = np.zeros([width, height])  # Macierz kątów między słońcem a wektorem normalnym powierzchni
    for x in range(width):
        for y in range(height):
            # Określanie trójkąta w celu obliczenia wektora normalnego powierzchni
            main_point = np.array([x * distance, mapa[x][y], y * distance])  # Główny punkt trójkąta
            if x % 2 == 0 and x < width - 1:
                if y < height - 1:
                    second_point = np.array([x * distance, mapa[x][y + 1], distance * (y + 1)])
                    third_point = np.array([(x + 1) * distance, mapa[x + 1][y], y * distance])
                else:
                    second_point = np.array([x * distance, mapa[x][y - 1], distance * (y - 1)])
                    third_point = np.array([(x + 1) * distance, mapa[x + 1][y], y * distance])
            else:
                if y > 0:
                    second_point = np.array([x * distance, mapa[x][y - 1], (y - 1) * distance])
                    third_point = np.array([(x - 1) * distance, mapa[x - 1][y], y * distance])
                else:
                    second_point = np.array([x * distance, mapa[x][y + 1], (y + 1) * distance])
                    third_point = np.array([(x - 1) * distance, mapa[x - 1][y], y * distance])

            vector_to_sun = sun - main_point
            # Wektor normalny powierzchni. Prostopadły do powierzchni trójkąta
            normal = np.cross(second_point - main_point, third_point - main_point)
            # Obliczanie kąta między wektorem normalnym x wektorem słońca
            angle_sun_surface = m.degrees(np.arccos(
                np.clip(np.dot(normal, vector_to_sun) / (np.linalg.norm(normal) * np.linalg.norm(vector_to_sun)), -1, 1)
            ))
            angles[x][y] = angle_sun_surface
    return angles


def reshade_map(width, height, mapa, mapaHSV, matrixOfAngles, max_min_dif, minimum):
    angles = np.sort(np.reshape(matrixOfAngles, -1))
    minAngle = np.min(angles)
    maxAngle = np.max(angles)
    # Określanie stopnia przyciemnienia na podstawie odchyleń kąta.
    for x in range(width):
        for y in range(height):
            mapaHSV[x][y][0] = get_h_value(mapa[x][y], max_min_dif, minimum)
            # Normalizowanie kąta x dodatkowe obliczenia uśredniające wynik
            relative_lightening(angles, x, y, mapaHSV, matrixOfAngles)
            normalize_lightening(x, y, mapaHSV, matrixOfAngles, maxAngle, minAngle)
            mapaHSV[x][y] = gr.hsv2rgb(mapaHSV[x][y][0], mapaHSV[x][y][1], mapaHSV[x][y][2])


def normalize_lightening(x, y, mapaHSV, matrixOfAngles, maxAngle, minAngle):
    # Otrzymanie kąta w zakresie <-1,1>
    normalized = ((matrixOfAngles[x][y] - minAngle) / (maxAngle - minAngle)) * 2 - 1
    if normalized < 0:
        mapaHSV[x][y][1] = ((1 + normalized) + mapaHSV[x][y][1]) / 2
    else:
        mapaHSV[x][y][2] = ((1 - normalized) + mapaHSV[x][y][2]) / 2


def relative_lightening(angles, x, y, mapaHSV, matrixOfAngles):
    # Sprawdzenie jak bardzo odchylony jest kąt w stosunku do wszystkich kątów
    position = np.where(angles == matrixOfAngles[x][y])[0]
    position = position[0] / len(angles)
    # Określenie S i V na podstawie pozycji kąta
    div = position - POSITION_BORDER_LIMIT
    if div < 0:
        mapaHSV[x][y][1] = 1 - np.sin(matrixOfAngles[x][y]) * abs(div)
    else:
        mapaHSV[x][y][2] = 1 - np.sin(matrixOfAngles[x][y]) * abs(div)


def antialiase(mapa):
    original_map_size = len(mapa)
    new_map_size = int(original_map_size * antialiasing_multiplier)-antialiasing_multiplier
    amapa = createHSVmatrix(new_map_size, new_map_size)
    ax = 0
    ay = 0
    for x in range(original_map_size-1):
        for y in range(original_map_size-1):
            amapa[ax][ay] = mapa[x][y]

            amapa[ax+1][ay] = gr.linear_interpolation((mapa[x][y], mapa[x+1][y]), 0.25)
            amapa[ax+2][ay] = gr.linear_interpolation((mapa[x][y], mapa[x+1][y]), 0.5)
            amapa[ax+3][ay] = gr.linear_interpolation((mapa[x][y], mapa[x+1][y]), 0.75)

            amapa[ax][ay+1] = gr.linear_interpolation((mapa[x][y], mapa[x][y+1]), 0.25)
            amapa[ax][ay+2] = gr.linear_interpolation((mapa[x][y], mapa[x][y+1]), 0.5)
            amapa[ax][ay+3] = gr.linear_interpolation((mapa[x][y], mapa[x][y+1]), 0.75)

            amapa[ax+1][ay+1] = gr.linear_interpolation((mapa[x][y], mapa[x+1][y+1]), 0.25)
            amapa[ax+2][ay+2] = gr.linear_interpolation((mapa[x][y], mapa[x+1][y+1]), 0.5)
            amapa[ax+3][ay+3] = gr.linear_interpolation((mapa[x][y], mapa[x+1][y+1]), 0.75)

            amapa[ax+3][ay+1] = gr.linear_interpolation((amapa[ax][ay+3], amapa[ax+3][ay+3]), 1/3)
            amapa[ax+3][ay+2] = gr.linear_interpolation((amapa[ax][ay+3], amapa[ax+3][ay+3]), 2/3)

            amapa[ax+1][ay+3] = gr.linear_interpolation((amapa[ax+3][ay], amapa[ax+3][ay+3]), 1/3)
            amapa[ax+2][ay+3] = gr.linear_interpolation((amapa[ax+3][ay], amapa[ax+3][ay+3]), 2/3)

            amapa[ax+2][ay+1] = gr.linear_interpolation((amapa[ax][ay+1], amapa[ax+2][ay+3]), .5)
            amapa[ax+1][ay+2] = gr.linear_interpolation((amapa[ax+1][ay], amapa[ax+3][ay+2]), .5)

            ay += antialiasing_multiplier
        ax += antialiasing_multiplier
        ay = 0
    return amapa


if __name__ == '__main__':
    mapa, width, height, distance = loadMapPoints("map_data.txt")
    simple_map = simple_shading(mapa, width, height)
    drawMap(simple_map, "simpleMap.pdf")
    mapaVector = vectorShading(mapa, width, height, distance)
    drawMap(mapaVector, "vectorMap.pdf")
    plt.close()
