"""Interface Streamlit do sistema RAG local.

Execute com:  uv run streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from rag.config import SUPPORTED_EXTENSIONS, get_settings
from rag.engine import RagEngine
from rag.ingestion import Ingestor
from rag.pinecone_manager import PineconeManager

st.set_page_config(page_title="RAG Local", page_icon="📚", layout="wide")


# --------------------------------------------------------------------------- #
# Recursos cacheados
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Conectando ao Pinecone…")
def load_backend():
    """Inicializa os recursos compartilhados do backend (uma única vez por processo).

    Carrega as configurações do `.env`, conecta ao Pinecone, garante que o
    index exista e instancia o `Ingestor` (que por sua vez carrega o manifesto).
    O resultado é cacheado por `st.cache_resource`, então reruns do Streamlit
    reutilizam as mesmas instâncias.

    Returns:
        Tupla `(settings, pinecone, ingestor, created)`, onde `created` indica
        se o index foi criado nesta inicialização.

    Raises:
        RuntimeError: se alguma variável de ambiente obrigatória estiver ausente.
        ValueError: se o modelo de embedding configurado não tiver dimensão conhecida.
    """
    settings = get_settings()
    pinecone = PineconeManager(settings)
    created = pinecone.ensure_index()
    ingestor = Ingestor(settings, pinecone)
    return settings, pinecone, ingestor, created


def build_engine(top_k: int, cutoff: float, temperature: float) -> RagEngine:
    """Constrói um novo `RagEngine` com os parâmetros de consulta informados.

    Args:
        top_k: Quantidade de chunks recuperados por pergunta.
        cutoff: Score mínimo de similaridade; chunks abaixo são descartados (0 = sem filtro).
        temperature: Temperatura do LLM na geração da resposta.

    Returns:
        Engine recém-criado, com memória conversacional vazia.
    """
    settings, pinecone, _, _ = load_backend()
    return RagEngine(settings, pinecone, top_k=top_k, similarity_cutoff=cutoff, temperature=temperature)


def get_engine(top_k: int, cutoff: float, temperature: float) -> RagEngine:
    """Retorna o engine da sessão, reconstruindo-o se os parâmetros mudaram.

    O engine é guardado em `st.session_state` junto com a tupla de parâmetros
    usada para criá-lo. Se o usuário alterar qualquer slider, um novo engine é
    criado, o que também zera a memória da conversa.

    Args:
        top_k: Quantidade de chunks recuperados por pergunta.
        cutoff: Score mínimo de similaridade (0 = sem filtro).
        temperature: Temperatura do LLM.

    Returns:
        O `RagEngine` atual da sessão.
    """
    key = (top_k, cutoff, temperature)
    if st.session_state.get("engine_key") != key or "engine" not in st.session_state:
        st.session_state.engine = build_engine(top_k, cutoff, temperature)
        st.session_state.engine_key = key
    return st.session_state.engine


# --------------------------------------------------------------------------- #
# Inicialização
# --------------------------------------------------------------------------- #
try:
    settings, pinecone, ingestor, index_created = load_backend()
except Exception as exc:  # noqa: BLE001
    st.error(f"Não foi possível inicializar o sistema: {exc}")
    st.stop()

if index_created and not st.session_state.get("index_created_notified"):
    st.toast(f"Index '{settings.index_name}' criado no Pinecone.", icon="✅")
    st.session_state.index_created_notified = True

st.session_state.setdefault("messages", [])


# --------------------------------------------------------------------------- #
# Sidebar: documentos + configurações
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.title("📚 RAG Local")
    st.caption(f"LLM: `{settings.llm_model}` · Embedding: `{settings.embedding_model}`")

    # ---- Status do index ------------------------------------------------- #
    stats = pinecone.stats()
    c1, c2 = st.columns(2)
    c1.metric("Vetores no namespace", stats["namespace_vectors"])
    c2.metric("Docs indexados", len(ingestor.manifest.data))
    st.caption(f"Index `{settings.index_name}` · namespace `{settings.namespace}`")

    st.divider()

    # ---- Upload ---------------------------------------------------------- #
    st.subheader("Documentos")
    uploaded = st.file_uploader(
        "Adicionar arquivos",
        type=[ext.lstrip(".") for ext in sorted(SUPPORTED_EXTENSIONS)],
        accept_multiple_files=True,
        help=f"Os arquivos são salvos em `{settings.documents_dir}` e indexados em seguida.",
    )
    if st.button("📥 Carregar e indexar", use_container_width=True, disabled=not uploaded):
        for f in uploaded:
            ingestor.save_uploaded(f.name, f.getvalue())
        with st.status("Indexando documentos…", expanded=True) as status:
            result = ingestor.sync(progress=lambda m: st.write(m))
            status.update(label="Indexação concluída", state="complete", expanded=False)
        st.success(
            f"Novos: {len(result.indexed)} · Atualizados: {len(result.updated)} · "
            f"Ignorados: {len(result.skipped)} · Chunks: {result.total_chunks}"
        )
        if result.failed:
            for name, err in result.failed.items():
                st.error(f"{name}: {err}")
        st.rerun()

    if st.button("🔄 Sincronizar pasta local", use_container_width=True,
                 help="Indexa arquivos colocados manualmente na pasta de documentos."):
        with st.status("Sincronizando…", expanded=True) as status:
            result = ingestor.sync(progress=lambda m: st.write(m))
            status.update(label="Sincronização concluída", state="complete", expanded=False)
        st.info(
            f"Novos: {len(result.indexed)} · Atualizados: {len(result.updated)} · "
            f"Ignorados: {len(result.skipped)}"
        )
        if result.failed:
            for name, err in result.failed.items():
                st.error(f"{name}: {err}")
        st.rerun()

    # ---- Lista de documentos --------------------------------------------- #
    docs = ingestor.list_local_documents()
    with st.expander(f"Arquivos na pasta ({len(docs)})", expanded=False):
        if not docs:
            st.caption("Nenhum documento ainda. Faça upload acima.")
        for path in docs:
            entry = ingestor.manifest.get(path.name)
            status_icon = "🟢" if entry else "⚪"
            chunks = f"{entry['chunks']} chunks" if entry else "não indexado"
            col_a, col_b = st.columns([5, 1])
            col_a.markdown(f"{status_icon} **{path.name}**  \n<small>{chunks}</small>",
                           unsafe_allow_html=True)
            if col_b.button("🗑️", key=f"del_{path.name}", help="Remover arquivo e seus vetores"):
                ingestor.remove_document(path.name)
                st.rerun()

    st.divider()

    # ---- Parâmetros de consulta ----------------------------------------- #
    st.subheader("Parâmetros")
    top_k = st.slider("Top-K (chunks recuperados)", 1, 15, settings.top_k)
    cutoff = st.slider("Similaridade mínima", 0.0, 1.0, 0.0, 0.05,
                       help="0 = sem filtro. Chunks abaixo deste score são descartados.")
    temperature = st.slider("Temperatura do LLM", 0.0, 1.0, settings.temperature, 0.05)

    st.divider()

    # ---- Ações ----------------------------------------------------------- #
    if st.button("🧹 Limpar conversa", use_container_width=True):
        st.session_state.messages = []
        if "engine" in st.session_state:
            st.session_state.engine.reset()
        st.rerun()

    # A reindexação apaga o namespace inteiro e o repovoa a partir da pasta local.
    # Isso só é reversível onde os documentos originais existem em disco: num
    # deploy a pasta vem vazia, então o reset destruiria o index sem recuperação.
    # Por isso a seção fica oculta salvo opt-in explícito via ALLOW_INDEX_RESET.
    if settings.allow_index_reset:
        with st.expander("Zona de perigo"):
            st.caption("Apaga todos os vetores do namespace e o manifesto. Os arquivos locais são mantidos.")
            if st.button("Reindexar tudo do zero", type="primary", use_container_width=True):
                ingestor.reset()
                with st.status("Reindexando…", expanded=True) as status:
                    ingestor.sync(progress=lambda m: st.write(m))
                    status.update(label="Reindexação concluída", state="complete", expanded=False)
                st.session_state.pop("engine", None)
                st.rerun()


# --------------------------------------------------------------------------- #
# Área principal: chat
# --------------------------------------------------------------------------- #
st.header("Converse com seus documentos")

if stats["namespace_vectors"] == 0:
    st.info("O index está vazio. Adicione documentos na barra lateral para começar.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📎 Fontes ({len(msg['sources'])})"):
                for i, src in enumerate(msg["sources"], 1):
                    page = f" · pág. {src['page']}" if src.get("page") else ""
                    score = f" · score {src['score']}" if src.get("score") is not None else ""
                    st.markdown(f"**{i}. {src['file_name']}**{page}{score}")
                    st.caption(src["text"][:600] + ("…" if len(src["text"]) > 600 else ""))

question = st.chat_input("Faça uma pergunta sobre os documentos…")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            engine = get_engine(top_k, cutoff, temperature)
            answer_text = st.write_stream(engine.stream(question))
            sources = [
                {"file_name": s.file_name, "page": s.page, "score": s.score, "text": s.text}
                for s in engine.last_sources
            ]
            if sources:
                with st.expander(f"📎 Fontes ({len(sources)})"):
                    for i, src in enumerate(sources, 1):
                        page = f" · pág. {src['page']}" if src.get("page") else ""
                        score = f" · score {src['score']}" if src.get("score") is not None else ""
                        st.markdown(f"**{i}. {src['file_name']}**{page}{score}")
                        st.caption(src["text"][:600] + ("…" if len(src["text"]) > 600 else ""))
        except Exception as exc:  # noqa: BLE001
            answer_text = f"Erro ao gerar resposta: {exc}"
            sources = []
            st.error(answer_text)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer_text, "sources": sources}
    )
