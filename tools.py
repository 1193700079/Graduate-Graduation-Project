import cv2
import numpy as np
def model_structure(model):
    blank = ' '
    print('-' * 90)
    print('|' + ' ' * 11 + 'weight name' + ' ' * 10 + '|' \
          + ' ' * 15 + 'weight shape' + ' ' * 15 + '|' \
          + ' ' * 3 + 'number' + ' ' * 3 + '|')
    print('-' * 90)
    num_para = 0
    type_size = 1  ##如果是浮点数就是4

    for index, (key, w_variable) in enumerate(model.named_parameters()):
        if len(key) <= 30:
            key = key + (30 - len(key)) * blank
        shape = str(w_variable.shape)
        if len(shape) <= 40:
            shape = shape + (40 - len(shape)) * blank
        each_para = 1
        for k in w_variable.shape:
            each_para *= k
        num_para += each_para
        str_num = str(each_para)
        if len(str_num) <= 10:
            str_num = str_num + (10 - len(str_num)) * blank

        print('| {} | {} | {} |'.format(key, shape, str_num))
    print('-' * 90)
    print('The total number of parameters: ' + str(num_para))
    print('The parameters of Model {}: {:4f}M'.format(model._get_name(), num_para * type_size / 1000 / 1000))
    print('-' * 90)
    


def scale_aligned_short(img, short_size=736):
    h, w = img.shape[0:2]
    scale = short_size * 1.0 / min(h, w)
    h = int(h * scale + 0.5)
    w = int(w * scale + 0.5)
    if h % 32 != 0:
        h = h + (32 - h % 32)
    if w % 32 != 0:
        w = w + (32 - w % 32)
    img = cv2.resize(img, dsize=(w, h))
    return img


def calculate_angle_with_x_axis(points):
    """
    计算矩形底边与x轴的夹角（以度为单位）。
    
    参数:
    points -- 矩形的四个顶点坐标的列表，每个顶点为(x, y)形式。
    
    返回:
    底边与x轴的夹角（度）。
    """
    # 按照y坐标排序，选取y坐标最小的两个点作为底部顶点
    bottom_points = sorted(points, key=lambda point: point[1])[:2]
    
    # 确定两点，确保从左到右（x1 < x2）
    p1, p2 = sorted(bottom_points)
    
    # 计算夹角
    dy = p2[1] - p1[1]
    dx = p2[0] - p1[0]
    
    # 使用atan2计算夹角，确保正确处理dx为0的情况
    angle_radians = np.arctan2(dy, dx)
    
    # 将弧度转换为度
    angle_degrees = np.degrees(angle_radians)
    
    return angle_degrees

def calculate_center(points):
    """
    计算矩形中心点的坐标。
    
    参数:
    points -- 矩形的四个顶点坐标的列表，每个顶点为(x, y)形式。
    
    返回:
    中心点的坐标。
    """
    # 计算所有x坐标和y坐标的平均值
    x_mean = sum(point[0] for point in points) / len(points)
    y_mean = sum(point[1] for point in points) / len(points)
    
    return (x_mean, y_mean)

def rotate_point(point, angle_deg, origin):
    """
    旋转一个点，围绕给定的原点旋转指定的角度。
    
    参数:
    - point: 旋转前的点坐标，格式为(x, y)。
    - angle_deg: 逆时针旋转的角度，单位为度。
    - origin: 旋转的原点坐标，格式为(x, y)。
    
    返回:
    - 旋转后的点坐标，格式为(x, y)。
    """
    # 将角度转换为弧度
    angle_rad = np.radians(angle_deg)
    
    # 旋转矩阵
    rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                                [np.sin(angle_rad), np.cos(angle_rad)]])
    
    # 将点和原点转换为numpy数组
    point = np.array(point)
    origin = np.array(origin)
    
    # 计算原点到点的向量
    point_vector = point - origin
    
    # 应用旋转矩阵
    rotated_point_vector = np.dot(rotation_matrix, point_vector)
    
    # 加回原点偏移以获得旋转后的坐标
    rotated_point = rotated_point_vector + origin
    
    return rotated_point

def rotate_rectangle(points, center, angle_degrees):
    """
    绕中心点旋转矩形。

    参数:
    points -- 矩形的四个顶点坐标的列表，每个顶点为(x, y)形式。
    center -- 旋转中心的坐标。
    angle_degrees -- 旋转角度（度）。

    返回:
    旋转后矩形顶点的新坐标列表。
    """
    return [rotate_point(px, py, center[0], center[1], angle_degrees) for px, py in points]


def find_max_enclosing_rectangle(polygon_vertices):
    # 提取多边形顶点的x和y坐标
    x_coords = [vertex[0] for vertex in polygon_vertices]
    y_coords = [vertex[1] for vertex in polygon_vertices]

    # 找到多边形在x和y方向上的最小和最大坐标值
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)

    # 计算矩形的宽度和高度
    width = max_x - min_x
    height = max_y - min_y

    # 计算矩形的面积
    area = width * height

    # 返回矩形的左下角和右上角顶点坐标及面积
    return (min_x, min_y), (max_x, max_y), area

# import jax.numpy as jnp
# def calculate_angle_with_x_axis_jax(points):
#     """
#     计算矩形底边与x轴的夹角（以度为单位）。
    
#     参数:
#     points -- 矩形的四个顶点坐标的列表，每个顶点为(x, y)形式。
    
#     返回:
#     底边与x轴的夹角（度）。
#     """
#     # 将输入点列表转换为JAX数组
#     points_jax = jnp.array(points)
    
#     # 按照y坐标排序，选取y坐标最小的两个点作为底部顶点
#     # 注意: JAX数组排序使用argsort和索引选择
#     indices_sorted_by_y = jnp.argsort(points_jax[:, 1])
#     bottom_points_indices = indices_sorted_by_y[:2]
#     bottom_points = points_jax[bottom_points_indices]
    
#     # 确定两点，确保从左到右（x1 < x2）
#     # JAX数组不支持列表式的sort函数，因此我们使用排序索引来确保顺序
#     indices_sorted_by_x = jnp.argsort(bottom_points[:, 0])
#     p1, p2 = bottom_points[indices_sorted_by_x]
    
#     # 计算夹角
#     dy = p2[1] - p1[1]
#     dx = p2[0] - p1[0]
    
#     # 使用atan2计算夹角，确保正确处理dx为0的情况
#     angle_radians = jnp.arctan2(dy, dx)
    
#     # 将弧度转换为度
#     angle_degrees = jnp.degrees(angle_radians)
    
#     return angle_degrees