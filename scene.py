import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def normalize(vector):
    return vector / np.linalg.norm(vector)


def sphere_intersect(center, radius, ray_origin, ray_direction):
    b = 2 * np.dot(ray_direction, ray_origin - center)
    c = np.linalg.norm(ray_origin - center) ** 2 - radius ** 2
    delta = b ** 2 - 4 * c
    if delta > 0:
        t1 = (-b + np.sqrt(delta)) / 2
        t2 = (-b - np.sqrt(delta)) / 2
        if t1 > 0 and t2 > 0:
            return min(t1, t2)
    return None


def nearest_intersected_object(objects, ray_origin, ray_direction):
    distances = [sphere_intersect(obj['center'], obj['radius'], ray_origin, ray_direction) for obj in objects]
    nearest_object = None
    min_distance = np.inf
    for index, distance in enumerate(distances):
        if distance and distance < min_distance:
            min_distance = distance
            nearest_object = objects[index]
    return nearest_object, min_distance


def stationary_sphere(center, radius, ambient_color, diffuse_color, specular_color, shininess):
    return {'center': np.array([center[0], center[1], center[2]]),
            'radius': radius,
            'ambient': np.array([ambient_color[0], ambient_color[1], ambient_color[2]]),
            'diffuse': np.array([diffuse_color[0], diffuse_color[1], diffuse_color[2]]),
            'specular': np.array([specular_color[0], specular_color[1], specular_color[2]]),
            'shininess': shininess}

def revolving_sphere(host, distance, radius, angle, ambient_color, diffuse_color, specular_color, shininess):
    host_center, host_radius = host['center'], host['radius']

    return {'center': np.array([host_center[0] + (host_radius + radius + distance) * np.sin(angle), host_center[1],
                                host_center[2] + (host_radius + radius + distance) * np.cos(angle)]),
            'radius': radius,
            'ambient': np.array([ambient_color[0], ambient_color[1], ambient_color[2]]),
            'diffuse': np.array([diffuse_color[0], diffuse_color[1], diffuse_color[2]]),
            'specular': np.array([specular_color[0], specular_color[1], specular_color[2]]),
            'shininess': shininess}


animation_frames=[]
frames = 1
width = 1920
height = 1080

camera = np.array([0, 0, 1])
ratio = float(width) / height
screen = (-1, 1 / ratio, 1, -1 / ratio)  # left, top, right, bottom

for index, theta in enumerate(np.linspace(0, 12 * np.pi, frames)):
    red_sphere = stationary_sphere([-0.2, 0, -1], 0.5, [0.1, 0, 0], [0.7, 0, 0], [1, 1, 1], 100)
    green_sphere = revolving_sphere(red_sphere, 0.5, 0.2, theta, [0, 0.1, 0], [0, 0.6, 0], [1, 1, 1], 100)
    purple_sphere = revolving_sphere(green_sphere, 0.1, 0.1, 2*theta, [0.1, 0, 0.1], [0.7, 0, 0.7], [1, 1, 1], 100)
    objects = [red_sphere, green_sphere, purple_sphere]

    light = {'position': np.array([5 * np.sin(theta/6), 5, 5 * np.cos(theta/3)]), 'ambient': np.array([1, 1, 1]),
             'diffuse': np.array([1, 1, 1]), 'specular': np.array([1, 1, 1])}

    image = np.zeros((height, width, 3))
    for i, y in enumerate(np.linspace(screen[1], screen[3], height)):
        for j, x in enumerate(np.linspace(screen[0], screen[2], width)):
            pixel = np.array([x, y, 0])
            origin = camera
            direction = normalize(pixel - origin)

            # check for intersections
            nearest_object, min_distance = nearest_intersected_object(objects, origin, direction)
            if nearest_object is None:
                continue

            # compute intersection point between ray and nearest object
            intersection = origin + min_distance * direction

            normal_to_surface = normalize(intersection - nearest_object['center'])
            shifted_point = intersection + 1e-5 * normal_to_surface
            intersection_to_light = normalize(light['position'] - shifted_point)

            _, min_distance = nearest_intersected_object(objects, shifted_point, intersection_to_light)
            intersection_to_light_distance = np.linalg.norm(light['position'] - intersection)
            is_shadowed = min_distance < intersection_to_light_distance

            if is_shadowed:
                continue

            # RGB
            illumination = np.zeros((3))

            # ambiant
            illumination += nearest_object['ambient'] * light['ambient']

            # diffuse
            illumination += nearest_object['diffuse'] * light['diffuse'] * np.dot(intersection_to_light,
                                                                                  normal_to_surface)

            # specular
            intersection_to_camera = normalize(camera - intersection)
            H = normalize(intersection_to_light + intersection_to_camera)
            illumination += nearest_object['specular'] * light['specular'] * np.dot(normal_to_surface, H) ** (
                    nearest_object['shininess'] / 4)

            image[i, j] = np.clip(illumination, 0, 1)

    plt.imsave('frames/frame%i.png' % (index), image)
    animation_frames.append(Image.open('frames/frame%i.png' % (index)))
    print("%i of %f rendered" % (index, frames.__int__()))
print("Done!")
#Generates animation, 25fps (40 ms per frame)
animation_frames[0].save('pillow_imagedraw.gif',
               save_all=True, append_images=animation_frames[1:], optimize=False, duration=40, loop=0)

