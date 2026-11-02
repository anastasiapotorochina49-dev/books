import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import hashlib

# --- НАСТРОЙКА СТРАНИЦЫ ---
st.set_page_config(page_title="Книжный рекомендатор", layout="wide")

# --- CSS СТИЛИ ---
st.markdown("""
    <style>
     @import url('https://fonts.googleapis.com/css2?family=Caveat:wght@400..700&family=Oswald:wght@200..700&family=Roboto+Slab:wght@100..900&family=Tiny5+Duo:ital,wght@0,100..900;1,100..900&family=Xmind:ital,wght@0,100..900;1,100..900&display=swap');
    
    /* 2. ПРИНУДИТЕЛЬНО ПРИМЕНЯЕМ ШРИФТ КО ВСЕМУ */
    /* Важно использовать !important и перечислять все теги, иначе Streamlit перебьет стиль */
    html, body, [class*="css"], .stMarkdown, p, div, h1, h2, h3, h4, h5, h6, li, span, button {
        font-family: 'Montserrat', sans-serif !important;
    }
    
    /* 3. УВЕЛИЧИВАЕМ РАЗМЕР ТЕКСТА ДЛЯ ЧИТАБЕЛЬНОСТИ */
    /* Основной текст (описания, параграфы) */
    p, .stMarkdown p {
        font-size: 1.15rem !important; 
        line-height: 1.6 !important;
        color: #231815 !important;
    }
    
    /* Палитра */
    :root {
        --blue_bg: #427AA2; /* Нежно-голубой фон */
        --red: #4D0E12;     /* Sceptre Red для кнопки */
        --blue: #A5BCD6;    /* Cerulean Blue для акцентов */
        --brown: #231815;   /* Java Brown для текста (очень темный) */
        --light_card: #ffffff;
    }
    
    /* Фон страницы - нежно-голубой */
    .stApp {
        background-color: var(--blue_bg);
    }
    
    /* Заголовки - цвет темного шоколада */
    h1, h2, h3 {
        color: var(--brown) !important;
        font-weight: 600 !important;
    }
    
    /* Карточки книг */
    .book-card {
        background-color: var(--light_card);
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        border-left: 5px solid var(--blue);
    }
    
    .book-avatar {
        width: 45px;
        height: 45px;
        border-radius: 50%;
        display: flex;
        justify-content: center;
        align-items: center;
        font-size: 18px;
        font-weight: 600;
        color: white;
        margin-right: 15px;
        flex-shrink: 0;
    }
    
    .book-title {
        font-size: 1rem;
        font-weight: 600;
        color: var(--brown);
        margin: 0;
    }
    .book-author {
        font-size: 0.85rem;
        color: #6b7280;
        margin-top: 2px;
    }
    
    /* Кнопка */
    .stButton>button {
        background-color: var(--red) !important;
        color: white !important;
        border-radius: 10px;
        border: none;
        padding: 10px 20px;
        font-weight: 500;
        width: 100%;
    }
        /* Делаем текст кнопки белым принудительно */
    .stButton>button p, 
    .stButton>button span, 
    .stButton>button div {
        color: white !important;
        font-weight: 600 !important;
    }
    
    /* Блок поиска */
    div[data-testid="stSelectbox"] {
        background-color: white;
        padding: 10px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #e5e7eb;
    }
    
        /* Картинка-баннер */
    .hero-image {
        width: 100%;
        height: 550px;
        object-fit: cover;
        border-radius: 20px;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
    }

    /* НОВЫЕ СТИЛИ ДЛЯ ЖЕЛТОГО БЛОКА */
    div[data-testid="stNotification"] {
        background-color: #F5EFC6 !important; 
        color: #231815 !important;           
        border: 1px solid #e5d8a3 !important; 
        border-radius: 10px;
    }
    
    div[data-testid="stNotification"] svg {
        fill: #231815 !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- ЗАГРУЗКА РЕСУРСОВ ---
@st.cache_resource
def load_resources():
    items = pd.read_csv('items_semantic.csv')
    index = faiss.read_index("books_faiss_semantic.index")
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    return items, index, model

with st.spinner("Загружаем нейросеть и базу книг..."):
    items, faiss_index, model = load_resources()

# --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ---
def get_palette_color(text):
    hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
    colors = ['#F5EFC6', '#4D0E12', '#A5BCD6', '#4A2E27', '#231815']
    color = colors[hash_val % len(colors)]
    text_color = "#231815" if color in ['#F5EFC6', '#A5BCD6'] else "#ffffff"
    return color, text_color

def render_book_card(title, author):
    color, text_color = get_palette_color(title)
    first_letter = title[0].upper() if title else "?"
    st.markdown(f"""
    <div class="book-card">
        <div class="book-avatar" style="background-color: {color}; color: {text_color};">{first_letter}</div>
        <div class="book-info">
            <p class="book-title">{title}</p>
            <p class="book-author">Автор: {author}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

def search_similar_books(query_title, top_n=5):
    matches = items[items['title'].str.contains(query_title, case=False, na=False)]
    if matches.empty:
        return None, f"Книга '{query_title}' не найдена."
    idx = matches.index[0]
    actual_title = items.loc[idx, 'title']
    actual_author = items.loc[idx, 'authors']
    
    query_text = f"Название: {actual_title}. Автор: {actual_author}"
    query_vector = model.encode([query_text]).astype('float32')
    faiss.normalize_L2(query_vector)
    distances, indices = faiss_index.search(query_vector, top_n + 1)
    
    results = []
    for i in range(1, len(indices[0])):
        res_idx = indices[0][i]
        results.append((items.loc[res_idx, 'title'], items.loc[res_idx, 'authors']))
    return results, actual_title

# ==========================================
# ИНТЕРФЕЙС
# ==========================================

col_left, col_right = st.columns([1, 1], gap="large")

# --- ЛЕВАЯ КОЛОНКА: КАРТИНКА ---

with col_left:
    # Просто используем встроенную функцию Streamlit
    st.image("header.jpg", use_container_width=True)

# --- ПРАВАЯ КОЛОНКА: ПОИСК ---
with col_right:
    # ВАЖНО: добавляем отступ сверху, чтобы заголовок выровнялся с картинкой
    st.markdown("<div style='margin-top: 60px;'></div>", unsafe_allow_html=True)
    
    st.markdown("""
    <h1 style="color: #231815; margin-bottom: 5px;">
        <span style="font-size: 2.5rem; vertical-align: middle; margin-right: 10px;">📖</span>
        КНИЖНЫЙ РЕКОМЕНДАТОР
    </h1>
    """, unsafe_allow_html=True)
    
    st.write("Введите название книги, и мы найдем для вас похожие по смыслу, жанру и атмосфере.")
    
    # Блок поиска
    with st.container():
        selected_book = st.selectbox(
            "Начните вводить название:",
            options=items['title'].tolist(),
            index=None,
            placeholder="Например: Мастер и Маргарита"
        )
        search_btn = st.button("Найти похожие", use_container_width=True)

    # Вывод результатов поиска
    if search_btn and selected_book:
        with st.spinner('Ищем...'):
            recs, message = search_similar_books(selected_book)
            if recs is None:
                st.error(message)
            else:
                st.markdown(f"""
<div style="background-color: #F5EFC6; color: #231815; padding: 15px; border-radius: 10px; font-weight: 500; margin-bottom: 15px; border-left: 5px solid #4D0E12;">
    Похожие на «{message}»:
</div>
""", unsafe_allow_html=True)
                for title, author in recs:
                    render_book_card(title, author)
    elif search_btn and not selected_book:
        st.warning("Выберите книгу из списка.")

# ==========================================
# НИЖНИЙ БЛОК: ТОП КНИГ
# ==========================================
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")
st.subheader("Популярное сейчас")

top_books = items.sample(4)
cols = st.columns(4)
for i, (_, row) in enumerate(top_books.iterrows()):
    with cols[i]:
        render_book_card(row['title'], row['authors'])