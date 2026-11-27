import streamlit as st

from common import chat, rag_prompt, search

st.title("Поиск по регламентам АО «ХимПромСинтез»")
chunk_size = st.selectbox("Размер чанка (коллекция)", [200, 500, 1000], index=1)
k = st.slider("top-k", 1, 10, 3)
use_rag = st.checkbox("Отвечать по документам (RAG)", value=True)
question = st.text_input("Вопрос", placeholder="Какие суточные в командировке по России?")

if st.button("Спросить") and question:
    if not use_rag:
        with st.spinner("Генерация..."):
            st.markdown("### Ответ (без RAG)")
            st.write(chat(question))
    else:
        hits = search(question, chunk_size, k)
        with st.spinner("Генерация..."):
            answer = chat(rag_prompt(question, hits))
        st.markdown("### Ответ (с RAG)")
        st.write(answer)
        st.markdown("### Источники")
        for h in hits:
            st.markdown(f"**{h.payload['source']}** (doc_id={h.payload['doc_id']}) · sim={h.score:.3f}")
            st.text(h.payload["chunk_text"][:300])
