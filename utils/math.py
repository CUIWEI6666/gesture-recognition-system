import numpy as np

def calculate_distance(p1, p2):
    """
    @brief 计算两个关键点之间的欧氏距离
    @param p1: 第一个关键点（带x,y,z属性）
    @param p2: 第二个关键点（带x,y,z属性）
    @return float: 两点之间的距离
    """
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

def calculate_angle(a, b, c):
    """
    @brief 计算三个关键点形成的夹角（以b为顶点）
    @param a: 第一个点
    @param b: 顶点
    @param c: 第三个点
    @return float: 夹角（角度制）
    """
    # 向量计算
    v1 = np.array([a.x - b.x, a.y - b.y])
    v2 = np.array([c.x - b.x, c.y - b.y])
    # 计算夹角
    cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cosine_angle))