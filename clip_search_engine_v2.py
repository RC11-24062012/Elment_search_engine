import torch
from PIL import Image
import os
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import numpy as np
from typing import List, Dict
import glob
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
import pickle
import time
import threading

# 初始化FastAPI应用
app = FastAPI(title="CLIP图文搜索引擎")

# 设置静态文件和模板目录
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# 加载CLIP模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = model.to(device)

BASE_DIR = os.path.dirname(__file__)
IMAGE_ROOT_DIR = os.path.join(BASE_DIR, "Element_image")
CACHE_DIR = os.path.join(BASE_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

current_area = None
image_features_dict = {}
image_paths = []

def get_available_areas():
    return [d for d in os.listdir(IMAGE_ROOT_DIR) if os.path.isdir(os.path.join(IMAGE_ROOT_DIR, d))]

def get_cache_file_path(area_name):
    return os.path.join(CACHE_DIR, f"clip_features_{area_name}.pkl")

def get_image_files(area_name):
    image_paths = set()
    area_dir = os.path.join(IMAGE_ROOT_DIR, area_name)
    extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
    for ext in extensions:
        image_paths.update(glob.glob(os.path.join(area_dir, f"*{ext}")))
        image_paths.update(glob.glob(os.path.join(area_dir, f"*{ext.upper()}")))
    return sorted(list(image_paths))

def save_features_cache(features_dict, cache_file):
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    with open(cache_file, 'wb') as f:
        pickle.dump(features_dict, f)
    print(f"✅ 缓存已保存: {cache_file}")

def load_features_cache(cache_file):
    try:
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    except:
        return None

def check_cache_validity(cache_file, image_paths):
    if not os.path.exists(cache_file):
        return False
    cached = load_features_cache(cache_file)
    return cached is not None and set(cached.keys()) == set(image_paths)

def normalize_features(features):
    return features / np.linalg.norm(features, axis=1, keepdims=True)

def extract_image_features(model, processor, image_path, device):
    try:
        image = Image.open(image_path).convert('RGB')
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
            features = image_features.cpu().numpy()
            return normalize_features(features)
    except:
        return None

def extract_text_features(model, processor, text, device):
    try:
        text = str(text).strip()
        if not text:
            raise ValueError("查询文本不能为空")
        inputs = processor(text=text, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            text_features = model.get_text_features(**inputs)
            return normalize_features(text_features)
    except:
        return None

def initialize_area_features(area_name):
    global image_features_dict, image_paths
    image_paths = get_image_files(area_name)
    cache_file = get_cache_file_path(area_name)

    if check_cache_validity(cache_file, image_paths):
        image_features_dict = load_features_cache(cache_file)
        return True

    print(f"🧠 正在处理区域: {area_name}")
    image_features_dict.clear()
    for img_path in tqdm(image_paths, desc=f"区域 {area_name}"):
        features = extract_image_features(model, processor, img_path, device)
        if features is not None:
            image_features_dict[img_path] = features.flatten()
    save_features_cache(image_features_dict, cache_file)
    return True

def initialize_all_areas():
    os.makedirs(CACHE_DIR, exist_ok=True)
    areas = get_available_areas()
    for area in areas:
        initialize_area_features(area)

def auto_refresh_loop(interval=300):  # 每5分钟检查一次
    while True:
        print("🔄 自动刷新中...")
        initialize_all_areas()
        time.sleep(interval)

@app.on_event("startup")
async def startup_event():
    if not os.path.exists(IMAGE_ROOT_DIR):
        print(f"❌ 图片目录不存在: {IMAGE_ROOT_DIR}")
        return
    if not os.path.exists("templates"):
        print("❌ templates 目录不存在")
        return
    if not os.path.exists("static"):
        os.makedirs("static")
    print("🚀 搜索引擎启动，自动缓存线程开启")
    threading.Thread(target=auto_refresh_loop, daemon=True).start()

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    try:
        areas = get_available_areas()
        return templates.TemplateResponse("index.html", {"request": request, "areas": areas})
    except Exception as e:
        return HTMLResponse(content=f"错误: {e}", status_code=500)

@app.post("/select_area")
async def select_area(area_name: str = Form(...)):
    global current_area
    if area_name not in get_available_areas():
        return {"error": "区域不存在"}
    current_area = area_name
    initialize_area_features(area_name)
    return {"message": f"已选择区域: {area_name}"}

@app.post("/search")
async def search_images(query: str = Form(...), top_k: int = Form(default=50)):
    global current_area
    if not current_area:
        return {"error": "请先选择区域"}
    text_features = extract_text_features(model, processor, query, device)
    if text_features is None:
        return {"error": "无法提取文本特征"}
    text_features = text_features.flatten()
    similarities = {
        img_path: float(np.dot(text_features, img_feat))
        for img_path, img_feat in image_features_dict.items()
        if img_feat.shape == text_features.shape
    }
    sorted_imgs = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return {"results": [{"image_path": p, "similarity": s, "filename": os.path.basename(p)} for p, s in sorted_imgs]}

@app.get("/image/{filename}")
async def get_image(filename: str):
    global current_area
    if not current_area:
        return {"error": "请先选择区域"}
    image_path = os.path.join(IMAGE_ROOT_DIR, current_area, filename)
    if os.path.exists(image_path):
        return FileResponse(image_path)
    return {"error": "图片不存在"}

if __name__ == "__main__":
    import sys
    host = "0.0.0.0" if "serve" in sys.argv else "127.0.0.1"
    uvicorn.run("clip_search_engine_v2:app", host=host, port=8000, reload=True)
