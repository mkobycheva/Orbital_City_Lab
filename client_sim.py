import socket
import json
import time
import math
import random
import osmnx as ox
import networkx as nx

SERVER_IP = "127.0.0.1"
SERVER_PORT = 5005
VEHICLE_ID = "Борт-101"

# Фіксована швидкість для стабільності
BASE_SPEED_KMH = 35.0 

print("⏳ Симулятор: Завантажую граф...")
G = ox.graph_from_place("Podilskyi District, Kyiv, Ukraine", network_type="drive")
nodes = list(G.nodes())

# 1. ФІКСОВАНИЙ МАРШРУТ (для стабільного демо)
start_lat, start_lon = 50.4646, 30.5165 
end_lat, end_lon = 50.4735, 30.4912 
origin = ox.distance.nearest_nodes(G, X=start_lon, Y=start_lat)
destination = ox.distance.nearest_nodes(G, X=end_lon, Y=end_lat)
route = nx.shortest_path(G, origin, destination, weight='length')
route_coords = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in route]

# Зберігаємо для дашборду
with open("route_info.json", "w") as f:
    json.dump({"path": [{"lat": lat, "lon": lon} for lat, lon in route_coords], 
               "start": {"lat": route_coords[0][0], "lon": route_coords[0][1]},
               "end": {"lat": route_coords[-1][0], "lon": route_coords[-1][1]}}, f)

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371000 
    return R * math.acos(math.sin(math.radians(lat1))*math.sin(math.radians(lat2)) + 
                         math.cos(math.radians(lat1))*math.cos(math.radians(lat2)) * math.cos(math.radians(lon2-lon1)))

def get_speed_for_turn(idx, route_coords):
    """Скидаємо швидкість, якщо попереду крутий поворот (наступні 2 вузли)"""
    if idx + 2 >= len(route_coords): return BASE_SPEED_KMH
    
    # Розрахунок кута між сегментами
    p1, p2, p3 = route_coords[idx], route_coords[idx+1], route_coords[idx+2]
    v1 = (p2[0]-p1[0], p2[1]-p1[1])
    v2 = (p3[0]-p2[0], p3[1]-p2[1])
    angle = abs(math.atan2(v2[1], v2[0]) - math.atan2(v1[1], v1[0]))
    
    if angle > 0.5: return 20.0 # Пригальмовуємо на повороті
    return BASE_SPEED_KMH

# Рух
current_idx = 0
current_lat, current_lon = route_coords[0]
seconds_elapsed = 0

print("🚗 Рухаємось...")
while current_idx < len(route_coords) - 1:
    target_lat, target_lon = route_coords[current_idx + 1]
    
    # Визначаємо швидкість
    current_speed_kmh = get_speed_for_turn(current_idx, route_coords)
    speed_mps = current_speed_kmh * (1000 / 3600)
    
    dist_to_target = calculate_distance(current_lat, current_lon, target_lat, target_lon)
    
    if dist_to_target <= speed_mps:
        current_lat, current_lon = target_lat, target_lon
        current_idx += 1
        continue
    
    ratio = speed_mps / dist_to_target
    current_lat += (target_lat - current_lat) * ratio
    current_lon += (target_lon - current_lon) * ratio
    
    # Відправка
    if not (15 <= (seconds_elapsed % 40) <= 30): # РЕБ кожні 40 сек
        packet = {"id": VEHICLE_ID, "lat": current_lat, "lon": current_lon, 
                  "speed": current_speed_kmh, "heading": 0} # heading обчислить сервер
        sock.sendto(json.dumps(packet).encode('utf-8'), (SERVER_IP, SERVER_PORT))
        print(f"🟢 Стан: {current_speed_kmh} км/год")
    else:
        print("🔴 РЕБ: зв'язок відсутній")
        
    seconds_elapsed += 1
    time.sleep(1)