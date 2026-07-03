import socket
import json
import time
import math
import numpy as np
import osmnx as ox
import networkx as nx
import joblib
import feature_utils

# ==========================================
# 1. КОНФІГУРАЦІЯ
# ==========================================
UDP_IP = "127.0.0.1"
UDP_PORT = 5005
TIMEOUT_THRESHOLD = 3.0  
STATE_FILE = "dashboard_state.json"

print("⏳ Завантаження графа доріг Києва (це займе хвилину)...")
G = ox.graph_from_place("Podilskyi District, Kyiv, Ukraine", network_type="drive")
print(f"✅ Граф завантажено: {len(G.nodes)} перехресть.")

print("⏳ Завантаження ML-моделі HistGradientBoostingRegressor...")
try:
    ai_model = joblib.load('model.pkl')
    print("✅ Справжню модель підключено успішно!")
except Exception as e:
    print(f"⚠️ Помилка завантаження моделі: {e}")
    ai_model = None

# ==========================================
# 2. ФІЗИКА ТА АЛГОРИТМИ
# ==========================================
def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371000 
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    a = math.sin(math.radians(lat2 - lat1)/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(math.radians(lon2 - lon1)/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def calculate_bearing(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(dlon))
    return (math.degrees(math.atan2(x, y)) + 360) % 360

def get_forward_node(lat, lon, heading):
    try:
        edge = ox.distance.nearest_edges(G, X=lon, Y=lat)
        u, v = edge[0], edge[1]
        
        lat_u, lon_u = G.nodes[u]['y'], G.nodes[u]['x']
        lat_v, lon_v = G.nodes[v]['y'], G.nodes[v]['x']
        
        diff_u = abs(calculate_bearing(lat, lon, lat_u, lon_u) - heading)
        diff_u = min(diff_u, 360 - diff_u)
        
        diff_v = abs(calculate_bearing(lat, lon, lat_v, lon_v) - heading)
        diff_v = min(diff_v, 360 - diff_v)
        
        return u if diff_u < diff_v else v
    except Exception:
        return ox.distance.nearest_nodes(G, X=lon, Y=lat)

def predict_next_graph_node(current_node, heading):
    neighbors = list(G.successors(current_node))
    if not neighbors: return current_node
    
    curr_lat, curr_lon = G.nodes[current_node]['y'], G.nodes[current_node]['x']
    best_node, min_angle_diff = neighbors[0], 360

    for neighbor in neighbors:
        n_lat, n_lon = G.nodes[neighbor]['y'], G.nodes[neighbor]['x']
        bearing = calculate_bearing(curr_lat, curr_lon, n_lat, n_lon)
        diff = abs(bearing - heading)
        diff = min(diff, 360 - diff)
        if diff < min_angle_diff:
            min_angle_diff = diff
            best_node = neighbor
    return best_node

def get_ai_speed_prediction(vehicle_state):
    speed_buffer = vehicle_state['speed_buffer']
    current_speed = speed_buffer[-1] if speed_buffer else 0.0
    
    if len(speed_buffer) < 5 or not ai_model:
        return current_speed

    try:
        features_1d = feature_utils.build_features(speed_buffer[-5:])
        predicted_speed = float(ai_model.predict([features_1d])[0])
        
        # ЗАПОБІЖНИК: Згладжуємо прогноз і не даємо машині зупинитися (мінімум 15 км/год)
        predicted_speed = np.clip(predicted_speed, current_speed - 2.0, current_speed + 2.0)
        return max(15.0, predicted_speed)
    except Exception:
        return current_speed

# ==========================================
# 3. ОСНОВНИЙ ЦИКЛ СЕРВЕРА
# ==========================================
vehicles = {}
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.settimeout(1.0)

print(f"🚀 Сервер запущено. Слухаю порт {UDP_PORT}...")

while True:
    current_time = time.time()
    
    # --- ПРИЙОМ ДАНИХ (ОНЛАЙН) ---
    try:
        data, addr = sock.recvfrom(1024)
        packet = json.loads(data.decode('utf-8'))
        v_id = packet['id']
        
        if v_id not in vehicles:
            vehicles[v_id] = {'speed_buffer': []}
            
        vehicles[v_id].update({
            'last_seen': current_time, 
            'lat': packet['lat'],
            'lon': packet['lon'], 
            'heading': packet['heading'], 
            'is_predicted': False
        })
        
        if 'target_node' in vehicles[v_id]:
            del vehicles[v_id]['target_node']
            
        vehicles[v_id]['speed_buffer'].append(packet['speed'])
        if len(vehicles[v_id]['speed_buffer']) > 5:
            vehicles[v_id]['speed_buffer'].pop(0)

        print(f"[🟢 ONLINE] Авто {v_id} | Швидкість: {packet['speed']:.1f} км/год")

    except socket.timeout: pass
    except Exception: pass

    # --- АВТОНОМНИЙ РУХ (РЕБ) ---
    for v_id, state in vehicles.items():
        if current_time - state['last_seen'] > TIMEOUT_THRESHOLD:
            
            predicted_speed = get_ai_speed_prediction(state)
            
            # Якщо РЕБ щойно увімкнувся, визначаємо першу ціль
            if 'target_node' not in state:
                state['target_node'] = get_forward_node(state['lat'], state['lon'], state['heading'])
                
            target_node = state['target_node']
            current_lat, current_lon = state['lat'], state['lon']
            
            # Розраховуємо, скільки метрів маємо проїхати за цю секунду
            speed_mps = predicted_speed * (1000 / 3600)
            dist_to_move = speed_mps
            
            # МАГІЯ ЦИКЛУ: Проходимо всі мікро-вузли, поки не витратимо dist_to_move
            loop_counter = 0
            while dist_to_move > 0.1 and loop_counter < 10:
                loop_counter += 1
                target_lat = G.nodes[target_node]['y']
                target_lon = G.nodes[target_node]['x']
                
                dist_to_target = calculate_distance(current_lat, current_lon, target_lat, target_lon)
                
                if dist_to_target > dist_to_move:
                    # Рухаємось по відрізку і вичерпуємо дистанцію
                    ratio = dist_to_move / dist_to_target
                    current_lat += (target_lat - current_lat) * ratio
                    current_lon += (target_lon - current_lon) * ratio
                    dist_to_move = 0
                else:
                    # Ми доїхали до вузла, але дистанція ще лишилась!
                    current_lat, current_lon = target_lat, target_lon
                    dist_to_move -= dist_to_target
                    
                    next_node = predict_next_graph_node(target_node, state['heading'])
                    if next_node == target_node:
                        break # Тупик, зупиняємось
                    
                    # Синхронізуємо компас машини з новою вулицею
                    new_heading = calculate_bearing(current_lat, current_lon, G.nodes[next_node]['y'], G.nodes[next_node]['x'])
                    if new_heading != 0.0 or current_lat != G.nodes[next_node]['y']:
                        state['heading'] = new_heading
                        
                    target_node = next_node

            # Зберігаємо результати плавного руху
            state.update({
                'lat': current_lat, 
                'lon': current_lon,
                'target_node': target_node,
                'last_seen': current_time,
                'is_predicted': True
            })
            
            state['speed_buffer'].append(predicted_speed)
            if len(state['speed_buffer']) > 5: state['speed_buffer'].pop(0)

            print(f"[🔴 РЕБ] Авто {v_id} | Швидкість: {predicted_speed:.1f} км/год | Фізика: {speed_mps:.1f} м/с")

    with open(STATE_FILE, 'w') as f:
        json.dump(vehicles, f)