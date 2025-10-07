import cv2
import mediapipe as mp
import numpy as np
import random
import time

# --- STATE and GLOBAL SETUP ---
WARN, MENU, SELECT, DRAW, PONG, HOCKEY, SETTINGS = 0, 1, 2, 3, 4, 5, 6
APP_STATE = WARN 

# Res (Width, Height)
RES = {
    "144p": (256, 144), 
    "240p": (320, 240),
    "360p": (640, 360),
    "480p": (854, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080)
}
DEFAULT_RES = "720p" 
SCREEN_MODE = 'FULLSCREEN' 
H, W = 720, 1280 
cap = None 
hands = None 
WIN_NAME = 'Hand Gesture Arcade'

# Click handling
last_click = time.time() 
mouse_click_pos = (-1, -1) 
CLICK_COOLDOWN = 0.5

# Warning screen
WARN_START = time.time()
WARN_DUR = 3.0 

# Drawing App
drawing_mode = False 
draw_color = (0, 0, 255) 
canvas = None
brush_size = 5 
draw_colors = {} 

# Pong
BALL_RAD = 15
BALL_SPEED = 5
PADDLE_H = 15
PADDLE_W = 120 
PADDLE_Y_PLAYER = 550
PADDLE_Y_AI = 20
ball_x, ball_y, vx, vy, score, lives, game_over = 0, 0, 0, 0, 0, 3, False
AI_SPEED = 0.95 

# Air Hockey
puck_x, puck_y, pvx, pvy = 0, 0, 0, 0
puck_r = 20
mallet_r = 40
p_mallet_x, p_mallet_y = 0, 0
ai_mallet_x, ai_mallet_y = 0, 0
p_score_ah, ai_score_ah = 0, 0
friction = 0.99 
GOAL_W = 0 
GOAL_H = 10 

# UI boxes 
MENU_BOXES = {} 
SELECT_BOXES = {}
MODE_BOX = {} 
ERASE_BOX = {} 
SCREEN_MODE_BOX = {}
ENTER_BOX = {} 


# --- CORE FUNCTIONS ---

def mouse_callback(event, x, y, flags, param):
    global mouse_click_pos, last_click
    if event == cv2.EVENT_LBUTTONDOWN:
        if time.time() - last_click > CLICK_COOLDOWN:
            mouse_click_pos = (x, y)
            last_click = time.time() 

def check_click(tip_x, tip_y, buttons):
    global last_click, mouse_click_pos
    
    click_x, click_y = -1, -1
    
    if tip_x != -1:
        click_x, click_y = tip_x, tip_y
    elif mouse_click_pos != (-1, -1):
        click_x, click_y = mouse_click_pos
    
    mouse_click_pos = (-1, -1) 
    
    if click_x == -1: return None 
        
    current_time = time.time()
    if click_x == tip_x and current_time - last_click < 0.5: return None

    for key, box in buttons.items():
        sx, sy = box["start"]; ex, ey = box["end"]
        if sx < click_x < ex and sy < click_y < ey:
            if click_x == tip_x: last_click = current_time 
            return box.get("state") if "state" in box else key
    return None

def draw_button(image, text, start, end, color):
    start = (np.clip(start[0], 0, W), np.clip(start[1], 0, H))
    end = (np.clip(end[0], 0, W), np.clip(end[1], 0, H))
    
    cv2.rectangle(image, start, end, color, -1)
    
    w_btn = max(1, end[0] - start[0])
    h_btn = max(1, end[1] - start[1])
    
    scale = h_btn / 75.0 
    scale = max(0.4, min(scale, 2.0))
    thickness = max(1, int(scale * 2))
    
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0]
    
    if text_size[0] > w_btn * 0.9: 
        scale *= (w_btn * 0.9) / text_size[0]
        thickness = max(1, int(scale * 2))
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0]

    text_x = start[0] + (w_btn - text_size[0]) // 2
    text_y = start[1] + (h_btn + text_size[1]) // 2
    
    cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness)

def draw_text_centered(image, text, y_pos, scale=1.5, color=(255, 255, 255), thickness=3):
    thickness = max(1, int(scale * 1.5)) 
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0]
    text_x = (W - text_size[0]) // 2
    y_pos = np.clip(y_pos, 20, H - 20)
    cv2.putText(image, text, (text_x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)

def init_ui(w, h):
    global MENU_BOXES, SELECT_BOXES, MODE_BOX, ERASE_BOX, draw_colors, GOAL_W, SCREEN_MODE_BOX, ENTER_BOX, PADDLE_Y_PLAYER
    
    w_scale = w / 640.0
    h_scale = h / 480.0
    
    MENU_BOXES = {
        "PLAY": {"start": (int(220*w_scale), int(150*h_scale)), "end": (int(420*w_scale), int(220*h_scale)), "state": SELECT},
        "SETTINGS": {"start": (int(220*w_scale), int(280*h_scale)), "end": (int(420*w_scale), int(350*h_scale)), "state": SETTINGS},
        "QUIT": {"start": (int(220*w_scale), int(410*h_scale)), "end": (int(420*w_scale), int(480*h_scale)), "state": -1} 
    }

    SELECT_BOXES = {
        "DRAWING":      {"start": (int(50*w_scale), int(180*h_scale)), "end": (int(200*w_scale), int(280*h_scale)), "state": DRAW},
        "PONG":         {"start": (int(240*w_scale), int(180*h_scale)), "end": (int(390*w_scale), int(280*h_scale)), "state": PONG},
        "AIR HOCKEY":   {"start": (int(430*w_scale), int(180*h_scale)), "end": (int(580*w_scale), int(280*h_scale)), "state": HOCKEY},
        "BACK":         {"start": (max(1, int(20*w_scale)), max(1, int(20*h_scale))), "end": (int(120*w_scale), int(70*h_scale)), "state": MENU} 
    }

    MODE_BOX = {"start": (int(50*w_scale), int(400*h_scale)), "end": (int(200*w_scale), int(450*h_scale))} 
    ERASE_BOX = {"start": (int(470*w_scale), int(400*h_scale)), "end": (int(620*w_scale), int(450*h_scale))} 
    
    draw_colors = {
        "RED":    {"start": (int(220*w_scale), int(330*h_scale)), "end": (int(270*w_scale), int(380*h_scale)), "color": (0, 0, 255)},
        "GREEN":  {"start": (int(280*w_scale), int(330*h_scale)), "end": (int(330*w_scale), int(380*h_scale)), "color": (0, 255, 0)},
        "BLUE":   {"start": (int(340*w_scale), int(330*h_scale)), "end": (int(390*w_scale), int(380*h_scale)), "color": (255, 0, 0)},
        "YELLOW": {"start": (int(400*w_scale), int(330*h_scale)), "end": (int(450*w_scale), int(380*h_scale)), "color": (0, 255, 255)},
    }
    
    button_w = int(200 * w_scale)
    button_h = int(50 * h_scale)
    SCREEN_MODE_BOX = {"start": (max(0, W // 2 - button_w // 2), max(0, int(H * 0.7))), 
                       "end": (min(W, W // 2 + button_w // 2), min(H, int(H * 0.7) + button_h))}

    enter_button_w = int(400 * w_scale)
    enter_button_h = int(100 * h_scale)
    ENTER_BOX = {"start": (max(0, W // 2 - enter_button_w // 2), max(0, int(H * 0.75) - enter_button_h // 2)), 
                 "end": (min(W, W // 2 + enter_button_w // 2), min(H, int(H * 0.75) + enter_button_h // 2))}

    GOAL_W = max(10, w // 3)
    PADDLE_Y_PLAYER = H - max(20, int(H * 0.05)) 

def set_res(res_key, attempt_dshow=True):
    global W, H, canvas, cap, BALL_SPEED
    
    target_w, target_h = RES.get(res_key, RES[DEFAULT_RES])
    
    if cap: cap.release()
    
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if attempt_dshow else cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Camera failed. Retrying simple mode.")
        if attempt_dshow: return set_res(res_key, attempt_dshow=False)
        else: print("CRITICAL: Camera unavailable."); cap = None; return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_h)
    
    success, img_temp = cap.read()
    if success and img_temp.size > 0:
        H, W, _ = img_temp.shape
        W, H = max(320, W), max(240, H)
    else:
        W, H = max(320, target_w), max(240, target_h)
        print(f"Warning: Defaulted to {W}x{H}")
    
    print(f"Set to: {W}x{H}. (Requested: {target_w}x{target_h})")
        
    init_ui(W, H)
    
    BALL_SPEED = max(3, int(W / 250 * 0.5))
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    
    reset_pong()
    reset_hockey_round()
    
    apply_screen_mode()

def apply_screen_mode():
    global SCREEN_MODE
    try:
        cv2.setWindowProperty(WIN_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN if SCREEN_MODE == 'FULLSCREEN' else cv2.WINDOW_NORMAL)
        if SCREEN_MODE == 'WINDOWED' and W > 300 and H > 200:
            cv2.resizeWindow(WIN_NAME, int(W * 0.8), int(H * 0.8))
    except cv2.error as e:
        print(f"Warning: Cannot apply screen mode: {e}")

# --- GAME/UI HELPERS ---

def reset_pong():
    global ball_x, ball_y, vx, vy, score, lives, game_over
    ball_x, ball_y = W // 2, H // 2
    vx = BALL_SPEED * random.choice([-1, 1])
    vy = BALL_SPEED * random.choice([-1, 1])
    score = 0
    lives = 3
    game_over = False

def reset_pong_ball():
    global ball_x, ball_y, vx, vy
    ball_x, ball_y = W // 2, H // 2
    vx = BALL_SPEED * random.choice([-1, 1])
    vy = BALL_SPEED * random.choice([-1, 1])

def reset_hockey_round(winner=None):
    global puck_x, puck_y, pvx, pvy, ai_mallet_x, ai_mallet_y
    puck_x, puck_y = W // 2, H // 2
    pvx, pvy = 0, 0
    ai_mallet_x, ai_mallet_y = W // 2, 50
    if winner == 'player': pvy = -5
    elif winner == 'ai': pvy = 5

# --- RENDER STATES ---

def render_warn(image, tip_x, tip_y):
    global APP_STATE, WARN_START
    image[:] = (0, 0, 0)
    
    scale = max(0.8, W / 1000.0) 
    
    draw_text_centered(image, "WARNING", int(H * 0.25), scale*2.5, (0, 0, 255))
    draw_text_centered(image, "Stop if dizzy or uncomfortable.", int(H * 0.45), scale*1.0)

    elapsed = time.time() - WARN_START
    remaining = WARN_DUR - elapsed

    if remaining > 0:
        countdown_text = f"READING: {max(0, int(remaining) + 1)}s"
        draw_text_centered(image, countdown_text, int(H * 0.75), scale*1.5, (0, 255, 255))
    else:
        draw_button(image, "ENTER", ENTER_BOX["start"], ENTER_BOX["end"], (0, 150, 0))
        if check_click(tip_x, tip_y, {"ENTER": ENTER_BOX}) == "ENTER":
            APP_STATE = MENU


def render_settings(image, tip_x, tip_y):
    global APP_STATE, SCREEN_MODE

    scale = max(0.8, W / 1000.0)
    
    draw_text_centered(image, "SETTINGS", int(H * 0.1), scale*1.5, (255, 255, 0))
    
    draw_button(image, "BACK", SELECT_BOXES["BACK"]["start"], SELECT_BOXES["BACK"]["end"], (100, 100, 100))
        
    draw_text_centered(image, f"Current: {W}x{H}", int(H * 0.2) + 10, scale*0.8, thickness=2)
    draw_text_centered(image, "Resolution:", int(H * 0.25), scale*1.0, thickness=2)
    
    res_keys = list(RES.keys())
    w_btn, h_btn = int(W / 6.5), int(H / 12)
    spacing = int(W / 60)
    
    x_start = W // 2 - (len(res_keys) * (w_btn + spacing) - spacing) // 2
    y_start = int(H * 0.35)
    
    res_buttons = {}
    for i, key in enumerate(res_keys):
        x, y = x_start + i * (w_btn + spacing), y_start
        start, end = (x, y), (x + w_btn, y + h_btn)
        res_buttons[key] = {"start": start, "end": end}
        color = (0, 255, 0) if W == RES[key][0] else (50, 50, 50)
        draw_button(image, key, start, end, color)

    draw_text_centered(image, "Screen Mode:", int(H * 0.6), scale*1.0, thickness=2)
    
    mode_text = f"Mode: {SCREEN_MODE}"
    mode_box = SCREEN_MODE_BOX
    draw_button(image, mode_text, mode_box["start"], mode_box["end"], (0, 150, 255))
    
    all_buttons = res_buttons.copy()
    all_buttons["BACK"] = SELECT_BOXES["BACK"]
    all_buttons["TOGGLE"] = mode_box
    
    clicked_action = check_click(tip_x, tip_y, all_buttons)

    if clicked_action == MENU: APP_STATE = MENU
    elif clicked_action in RES: set_res(clicked_action)
    elif clicked_action == "TOGGLE":
        SCREEN_MODE = 'WINDOWED' if SCREEN_MODE == 'FULLSCREEN' else 'FULLSCREEN'
        apply_screen_mode()

def render_menu(image, tip_x, tip_y):
    global APP_STATE
    draw_text_centered(image, "HAND ARCADE", int(H * 0.15), 1.5, (255, 255, 0))
    next_state = check_click(tip_x, tip_y, MENU_BOXES)
    for key, box in MENU_BOXES.items():
        draw_button(image, key, box["start"], box["end"], (0, 150, 255))
    if next_state is not None:
        if next_state == -1: return False
        APP_STATE = next_state
    return True

def render_select_game(image, tip_x, tip_y):
    global APP_STATE
    draw_text_centered(image, "SELECT GAME", int(H * 0.15), 1.5, (255, 255, 0))
    next_state = check_click(tip_x, tip_y, SELECT_BOXES)
    for key, box in SELECT_BOXES.items():
        draw_button(image, key, box["start"], box["end"], (0, 200, 100))
    if next_state is not None:
        APP_STATE = next_state
        if APP_STATE == PONG: reset_pong()
        elif APP_STATE == HOCKEY: reset_hockey_round()

def render_drawing(image, tip_x, tip_y, thumb_x, thumb_y):
    global APP_STATE, drawing_mode, canvas, draw_color, brush_size
    
    draw_button(image, "BACK", SELECT_BOXES["BACK"]["start"], SELECT_BOXES["BACK"]["end"], (100, 100, 100))
    mode_text = "ON" if drawing_mode else "OFF"
    mode_box_color = (0, 0, 255) if drawing_mode else (0, 255, 0)
    draw_button(image, mode_text, MODE_BOX["start"], MODE_BOX["end"], mode_box_color)
    draw_button(image, "ERASE", ERASE_BOX["start"], ERASE_BOX["end"], (0, 0, 255))
    for key, box in draw_colors.items(): cv2.rectangle(image, box["start"], box["end"], box["color"], -1)

    all_buttons = {**draw_colors, "BACK": SELECT_BOXES["BACK"], "MODE_TOGGLE": MODE_BOX, "ERASE": ERASE_BOX}
    clicked_action = check_click(tip_x, tip_y, all_buttons)

    if clicked_action == MENU: APP_STATE = SELECT; return image
    
    if clicked_action == "MODE_TOGGLE": drawing_mode = not drawing_mode
    if clicked_action == "ERASE": canvas = np.zeros((H, W, 3), dtype=np.uint8)
        
    if clicked_action in draw_colors: draw_color = draw_colors[clicked_action]["color"]
    
    if thumb_x != -1 and tip_x != -1:
        distance = np.linalg.norm(np.array([tip_x, tip_y]) - np.array([thumb_x, thumb_y]))
        brush_size = int(np.clip(distance / 5, 5, max(10, int(W/50)))) 
        cv2.putText(image, f"Size: {brush_size}", (W - 150, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    if drawing_mode and tip_x != -1:
        safe_x = np.clip(tip_x, brush_size, W - brush_size)
        safe_y = np.clip(tip_y, brush_size, H - brush_size)
        cv2.circle(canvas, (safe_x, safe_y), brush_size, draw_color, -1)
    
    img_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, img_mask = cv2.threshold(img_gray, 1, 255, cv2.THRESH_BINARY_INV)
    img_mask = cv2.cvtColor(img_mask, cv2.COLOR_GRAY2BGR)
    img_mask_inv = cv2.bitwise_not(img_mask)
    img_bg = cv2.bitwise_and(image, img_mask)
    img_fg = cv2.bitwise_and(canvas, img_mask_inv)
    return cv2.add(img_bg, img_fg)

def render_pong(image, tip_x, tip_y, wrist_x):
    global APP_STATE, ball_x, ball_y, vx, vy, score, lives, game_over
    global PADDLE_W, PADDLE_H, BALL_RAD, PADDLE_Y_PLAYER, PADDLE_Y_AI
    
    paddle_scale = W / 1280.0
    PADDLE_W = max(50, int(120 * paddle_scale))
    PADDLE_H = max(10, int(15 * paddle_scale))
    BALL_RAD = max(8, int(15 * paddle_scale))
    PADDLE_Y_AI = max(10, int(H * 0.05))
    
    draw_button(image, "BACK", SELECT_BOXES["BACK"]["start"], SELECT_BOXES["BACK"]["end"], (100, 100, 100))
    if check_click(tip_x, tip_y, {"BACK": SELECT_BOXES["BACK"]}) == MENU:
        APP_STATE = SELECT; return image

    p_center_x = np.clip(wrist_x, PADDLE_W // 2, W - PADDLE_W // 2)
    ai_p_x = W // 2 
    
    if not game_over:
        ai_p_x += (ball_x - ai_p_x) * (1 - AI_SPEED)
        ai_p_x = np.clip(ai_p_x, PADDLE_W // 2, W - PADDLE_W // 2)

        ball_x += vx; ball_y += vy
        
        if ball_x - BALL_RAD < 0 or ball_x + BALL_RAD > W: vx *= -1
        
        if ball_y + BALL_RAD >= PADDLE_Y_PLAYER:
            p_left = p_center_x - PADDLE_W // 2; p_right = p_center_x + PADDLE_W // 2
            if p_left < ball_x < p_right and ball_y < PADDLE_Y_PLAYER + PADDLE_H:
                vy *= -1; score += 1; vx += (ball_x - p_center_x) / 10; vx = np.clip(vx, -15, 15)
            elif ball_y + BALL_RAD > H: 
                lives -= 1
                if lives <= 0: game_over = True
                else: reset_pong_ball() 

        if ball_y - BALL_RAD <= PADDLE_Y_AI + PADDLE_H:
            ai_p_left = ai_p_x - PADDLE_W // 2; ai_p_right = ai_p_x + PADDLE_W // 2
            if ai_p_left < ball_x < ai_p_right and ball_y > PADDLE_Y_AI:
                vy *= -1; vx += (ball_x - ai_p_x) / 10; vx = np.clip(vx, -15, 15)
            elif ball_y - BALL_RAD < 0: 
                vy *= -1

    p_start = (p_center_x - PADDLE_W // 2, PADDLE_Y_PLAYER); p_end = (p_center_x + PADDLE_W // 2, PADDLE_Y_PLAYER + PADDLE_H)
    cv2.rectangle(image, p_start, p_end, (255, 0, 0), -1)
    ai_p_start = (int(ai_p_x) - PADDLE_W // 2, PADDLE_Y_AI); ai_p_end = (int(ai_p_x) + PADDLE_W // 2, PADDLE_Y_AI + PADDLE_H)
    cv2.rectangle(image, ai_p_start, ai_p_end, (0, 0, 255), -1)
    cv2.circle(image, (int(ball_x), int(ball_y)), BALL_RAD, (0, 0, 255), -1)

    cv2.putText(image, f"Score: {score}", (W - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(image, f"Lives: {lives}", (W // 2 - 50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    if game_over: draw_text_centered(image, "GAME OVER! R to Restart", H // 2, 1.5, (0, 0, 255), 3)

    return image

def render_air_hockey(image, tip_x, tip_y):
    global APP_STATE, puck_x, puck_y, pvx, pvy, p_mallet_x, p_mallet_y, ai_mallet_x, ai_score_ah, p_score_ah
    global puck_r, mallet_r, GOAL_H, GOAL_W
    
    scale = W / 1280.0
    puck_r = max(8, int(20 * scale))
    mallet_r = max(15, int(40 * scale))
    GOAL_H = max(5, int(10 * scale))
    GOAL_W = max(50, W // 3)

    draw_button(image, "BACK", SELECT_BOXES["BACK"]["start"], SELECT_BOXES["BACK"]["end"], (100, 100, 100))
    if check_click(tip_x, tip_y, {"BACK": SELECT_BOXES["BACK"]}) == MENU:
        APP_STATE = SELECT; return image
    
    p_mallet_x = np.clip(tip_x, mallet_r, W - mallet_r)
    p_mallet_y = np.clip(tip_y, H // 2, H - mallet_r)
    
    ai_mallet_x += (puck_x - ai_mallet_x) * 0.05
    ai_mallet_x = np.clip(ai_mallet_x, mallet_r, W - mallet_r)
    ai_mallet_y = np.clip(int(H * 0.1), mallet_r, H // 2 - mallet_r)

    puck_x += pvx; puck_y += pvy; pvx *= friction; pvy *= friction
    
    if puck_x - puck_r < 0 or puck_x + puck_r > W: pvx *= -1
    if puck_y - puck_r < 0 or puck_y + puck_r > H: pvy *= -1
        
    dist_player = np.linalg.norm(np.array([puck_x, puck_y]) - np.array([p_mallet_x, p_mallet_y]))
    if dist_player < puck_r + mallet_r:
        hit_direction = np.array([puck_x - p_mallet_x, puck_y - p_mallet_y])
        hit_direction = hit_direction / np.linalg.norm(hit_direction); pvx = hit_direction[0] * 10; pvy = hit_direction[1] * 10 
        
    dist_ai = np.linalg.norm(np.array([puck_x, puck_y]) - np.array([ai_mallet_x, ai_mallet_y]))
    if dist_ai < puck_r + mallet_r:
        hit_direction = np.array([puck_x - ai_mallet_x, puck_y - ai_mallet_y])
        hit_direction = hit_direction / np.linalg.norm(hit_direction); pvx = hit_direction[0] * 10; pvy = hit_direction[1] * 10 

    goal_x_center = W // 2; goal_x_start = goal_x_center - GOAL_W // 2; goal_x_end = goal_x_center + GOAL_W // 2
    
    if puck_y - puck_r < GOAL_H and goal_x_start < puck_x < goal_x_end: 
        p_score_ah += 1
        reset_hockey_round(winner='player')
        
    if puck_y + puck_r > H - GOAL_H and goal_x_start < puck_x < goal_x_end: 
        ai_score_ah += 1
        reset_hockey_round(winner='ai')
        
    cv2.line(image, (0, H // 2), (W, H // 2), (255, 255, 255), 1)
    
    goal_start_top = (goal_x_start, 0); goal_end_top = (goal_x_end, GOAL_H)
    cv2.rectangle(image, goal_start_top, goal_end_top, (0, 0, 255), -1)
    
    goal_start_bottom = (goal_x_start, H - GOAL_H); goal_end_bottom = (goal_x_end, H)
    cv2.rectangle(image, goal_start_bottom, goal_end_bottom, (255, 0, 0), -1)
    
    cv2.circle(image, (int(p_mallet_x), int(p_mallet_y)), mallet_r, (255, 0, 0), -1)
    cv2.circle(image, (int(ai_mallet_x), int(ai_mallet_y)), mallet_r, (0, 0, 255), -1)
    cv2.circle(image, (int(puck_x), int(puck_y)), puck_r, (0, 255, 255), -1)

    cv2.putText(image, f"You: {p_score_ah}", (W - 150, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(image, f"AI: {ai_score_ah}", (W - 150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return image


# ----------------------------------------------------------------------
# --- MAIN LOOP ---
# ----------------------------------------------------------------------

mp_hands = mp.solutions.hands 
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5) 

cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
cv2.setMouseCallback(WIN_NAME, mouse_callback)

set_res(DEFAULT_RES) 

running = True
while running:
    if cap is None or not cap.isOpened():
        print("Camera not running. Exiting.")
        running = False
        continue
        
    success, image = cap.read()
    if not success: continue

    if image.shape[0] != H or image.shape[1] != W:
        image = cv2.resize(image, (W, H))

    image = cv2.flip(image, 1)
    processed_image = image.copy()
    
    image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = hands.process(image_rgb)
    
    tip_x, tip_y, thumb_x, thumb_y, wrist_x = -1, -1, -1, -1, -1
    
    if results.multi_hand_landmarks:
        lm = results.multi_hand_landmarks[0].landmark
        
        tip_lm = lm[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        tip_x, tip_y = int(tip_lm.x * W), int(tip_lm.y * H)
        
        thumb_lm = lm[mp_hands.HandLandmark.THUMB_TIP]
        thumb_x, thumb_y = int(thumb_lm.x * W), int(thumb_lm.y * H)

        wrist_lm = lm[mp_hands.HandLandmark.WRIST]
        wrist_x, _ = int(wrist_lm.x * W), int(wrist_lm.y * H)
        
        if APP_STATE != WARN:
             cv2.circle(processed_image, (tip_x, tip_y), max(5, int(W/200)), (255, 255, 255), 2)

    final_image = processed_image

    if APP_STATE == WARN:
        render_warn(final_image, tip_x, tip_y)

    elif APP_STATE == MENU:
        if not render_menu(final_image, tip_x, tip_y):
            running = False
            
    elif APP_STATE == SELECT:
        render_select_game(final_image, tip_x, tip_y)
    
    elif APP_STATE == SETTINGS:
        render_settings(final_image, tip_x, tip_y)
        
    elif APP_STATE == DRAW:
        final_image = render_drawing(final_image, tip_x, tip_y, thumb_x, thumb_y)
        
    elif APP_STATE == PONG:
        final_image = render_pong(final_image, tip_x, tip_y, wrist_x)
        
    elif APP_STATE == HOCKEY:
        final_image = render_air_hockey(final_image, tip_x, tip_y)
        
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        running = False
    
    if APP_STATE == PONG and game_over and key == ord('r'):
        reset_pong()

    cv2.imshow(WIN_NAME, final_image)


# --- Cleanup ---
if cap is not None:
    cap.release()
cv2.destroyAllWindows()