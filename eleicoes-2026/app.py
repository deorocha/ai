# -*- coding: utf-8 -*-
"""
app.py — Dashboard Streamlit da projeção do 2º turno de 2026.

Rodar:
    streamlit run app.py

Dependências:
    pip install streamlit pandas numpy plotly geopandas geobr

Arquivos utilizados:
    ./style.css                       → estilos da interface
    ./resultados/previsao_2T_2026_estado.csv
    ./resultados/previsao_2T_2026_municipio.csv
"""

import os
import unicodedata
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

import streamlit_analytics2 as streamlit_analytics

# Inicia o rastreamento (recomenda-se configurar uma senha para proteger o dashboard)
with streamlit_analytics.track():
    st.title("🗳️ Projeção do 2º Turno — 2026")
    st.caption("Lula × Flávio Bolsonaro — swing de 2022 aplicado ao 1º turno de 2026")
    # O contador e os dados ficam visíveis adicionando "?analytics=on" na URL do seu app

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Eleições 2026 — 2º Turno",
    page_icon="🗳️",
    layout="wide",
)

# ------------------------------------------------------------------
# LOCALIZA A RAIZ DO PROJETO
# ------------------------------------------------------------------
def encontrar_raiz():
    p = os.path.dirname(os.path.abspath(__file__))
    for _ in range(6):
        if os.path.isdir(os.path.join(p, "resultados")):
            return p
        pai = os.path.dirname(p)
        if pai == p:
            break
        p = pai
    return os.path.dirname(os.path.abspath(__file__))

RAIZ = encontrar_raiz()
PASTA_RESULT = os.path.join(RAIZ, "resultados")

# ------------------------------------------------------------------
# CSS EXTERNO — carregado de style.css
# ------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def carregar_css(caminho):
    with open(caminho, "r", encoding="utf-8") as f:
        return f.read()


def injetar_css():
    candidatos = [
        os.path.join(RAIZ, "style.css"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "style.css"),
    ]
    for caminho in candidatos:
        if os.path.exists(caminho):
            css = carregar_css(caminho)
            st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
            return
    st.warning("Arquivo `style.css` não encontrado.")


injetar_css()

# ------------------------------------------------------------------
# CONSTANTES
# ------------------------------------------------------------------
COR_LULA   = "#d62728"  # vermelho
COR_FLAVIO = "#1f77b4"  # azul
COR_SEM    = "#d9d9d9"  # cinza

# Apenas 2 arquivos são necessários:
#   - estado.csv     → nível "Estadual"
#   - municipio.csv  → níveis "Geral (Brasil)" e "Municipal"
ARQUIVOS = {
    "Geral (Brasil)": ("previsao_2T_2026_municipio.csv", "nacional"),
    "Estadual":       ("previsao_2T_2026_estado.csv",    "estado"),
    "Municipal":      ("previsao_2T_2026_municipio.csv", "municipio"),
}

LIMITE_TABELA = {
    "nacional":  10000,
    "estado":    10000,
    "municipio": 20000,
}

# ------------------------------------------------------------------
# VERSÃO DO CACHE GEO.
# Mude este número SEMPRE que alterar o schema do shapefile para
# forçar o Streamlit a descartar caches antigos.
# ------------------------------------------------------------------
VERSAO_GEO = 4

# ------------------------------------------------------------------
# CARREGAMENTO DE RESULTADOS
# ------------------------------------------------------------------
@st.cache_data(show_spinner="Carregando resultados…")
def carregar(arquivo):
    caminho = os.path.join(PASTA_RESULT, arquivo)
    if not os.path.exists(caminho):
        return None
    return pd.read_csv(caminho, sep=";", decimal=",", encoding="utf-8-sig")

# ------------------------------------------------------------------
# CARREGAMENTO DE GEODADOS
# ------------------------------------------------------------------
@st.cache_resource(show_spinner="Carregando geodados de estados…")
def geo_estados(versao=VERSAO_GEO):
    import geobr
    gdf = geobr.read_state(year=2020)
    gdf = gdf[["abbrev_state", "name_state", "geometry"]].copy()
    gdf["abbrev_state"] = gdf["abbrev_state"].astype(str).str.upper()
    return gdf


@st.cache_resource(show_spinner="Carregando municípios do Brasil… (1ª vez pode demorar alguns minutos)")
def geo_municipios_brasil(versao=VERSAO_GEO):
    import geobr
    gdf = geobr.read_municipality(year=2020)
    gdf = gdf[["code_muni", "name_muni", "abbrev_state", "geometry"]].copy()
    gdf["abbrev_state"] = gdf["abbrev_state"].astype(str).str.upper()
    try:
        gdf["geometry"] = gdf["geometry"].simplify(0.02, preserve_topology=True)
    except Exception:
        pass
    return gdf


@st.cache_resource(show_spinner="Carregando municípios da UF…")
def geo_municipios_uf(uf, versao=VERSAO_GEO):
    import geobr
    gdf = geobr.read_municipality(code_muni=uf, year=2020)
    gdf = gdf[["code_muni", "name_muni", "abbrev_state", "geometry"]].copy()
    gdf["abbrev_state"] = gdf["abbrev_state"].astype(str).str.upper()
    try:
        gdf["geometry"] = gdf["geometry"].simplify(0.005, preserve_topology=True)
    except Exception:
        pass
    return gdf


# ------------------------------------------------------------------
# UTIL
# ------------------------------------------------------------------
def normaliza(s):
    return (unicodedata.normalize("NFKD", str(s))
            .encode("ASCII", "ignore").decode("ASCII")
            .lower().strip())


def adiciona_vencedor(df):
    df = df.copy()
    df["vencedor"] = np.where(
        df["share_lula_2T_prev"] >= df["share_flavio_2T_prev"],
        "Lula", "Flávio"
    )
    df["votos_venc"] = np.where(
        df["vencedor"] == "Lula", df["v_lula_2T_prev"], df["v_flavio_2T_prev"]
    )
    df["share_venc"] = np.where(
        df["vencedor"] == "Lula", df["share_lula_2T_prev"], df["share_flavio_2T_prev"]
    )
    return df


def tem_geo():
    try:
        import geobr        # noqa
        import geopandas    # noqa
        return True
    except Exception:
        return False


def fmt_int(x):
    if pd.isna(x):
        return "—"
    return f"{int(round(x)):,}".replace(",", ".")


def mapa_choropleth(gdf, col_id, nome_hover, cols_hover, titulo=None,
                    height=650, auto_zoom=False):
    """gdf DEVE ter coluna 'vencedor'. Colunas de hover devem existir.

    auto_zoom=False → usa o bounding box fixo do Brasil (para níveis
                       Nacional e Estadual).
    auto_zoom=True  → calcula o bounding box a partir dos próprios dados
                       (usado no nível Municipal, para enquadrar só a UF
                       selecionada).
    """
    geojson = gdf.set_index(col_id).__geo_interface__
    fig = px.choropleth(
        gdf,
        geojson=geojson,
        locations=col_id,
        color="vencedor",
        color_discrete_map={
            "Lula": COR_LULA,
            "Flávio": COR_FLAVIO,
            "Sem dados": COR_SEM,
        },
        hover_name=nome_hover,
        hover_data=cols_hover,
    )

    if auto_zoom:
        # Calcula o bounding box dos dados e aplica uma folga de 4%
        minx, miny, maxx, maxy = gdf.total_bounds
        dx = (maxx - minx) * 0.04
        dy = (maxy - miny) * 0.04
        lon_range = [minx - dx, maxx + dx]
        lat_range = [miny - dy, maxy + dy]
    else:
        # Bounding box fixo do Brasil (usado nos níveis Nacional e Estadual)
        lon_range = [-75, -33]
        lat_range = [-35, 6]

    fig.update_geos(
        visible=False,
        projection_type="mercator",
        lataxis_range=lat_range,
        lonaxis_range=lon_range,
        showcountries=False,
        showcoastlines=False,
        showland=False,
        showframe=False,
    )

    try:
        fig.update_traces(marker_line_width=0)
    except Exception:
        pass

    fig.update_layout(
        height=height,
        autosize=True,
        margin=dict(l=0, r=0, t=0, b=30),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.01,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
            title=None,
            itemsizing="constant",
        ),
    )
    return fig

# ------------------------------------------------------------------
# APP
# ------------------------------------------------------------------

# ---- SIDEBAR ----
st.sidebar.header("Abrangência")
abrangencia = st.sidebar.radio(
    "Nível de análise:",
    list(ARQUIVOS.keys()),
    index=1,
)

arquivo, chave = ARQUIVOS[abrangencia]
df = carregar(arquivo)

if df is None:
    st.error(
        f"Arquivo `{arquivo}` não encontrado em `{PASTA_RESULT}`. "
        "Rode primeiro o script `02_prever_2T_2026.py`."
    )
    st.stop()

df = adiciona_vencedor(df)

# Filtro por UF (apenas para o nível Municipal)
uf_sel = None
if chave == "municipio" and "SG_UF" in df.columns:
    ufs = sorted(df["SG_UF"].dropna().unique().tolist())
    escolha = st.sidebar.selectbox("Filtrar por UF:", ["Todas"] + ufs)
    if escolha != "Todas":
        uf_sel = escolha
        df = df[df["SG_UF"] == escolha]

st.sidebar.markdown("---")
st.sidebar.caption("Cores: 🔴 Lula · 🔵 Flávio Bolsonaro")

# ---- Expander: Metodologia ----
with st.sidebar.expander("📖 Metodologia da projeção", expanded=False):
    st.markdown(
        """
### Em uma frase
Aplicamos aos votos do 1º turno de 2026 a **variação (swing) que cada
candidato teve entre o 1º e o 2º turno de 2022**, calculada na mesma
geografia (país, estado ou município).

---

### Passo a passo

**1. Universo considerado**
- Apenas votos **válidos** (exclui brancos, nulos e abstenções).
- Cargo: **Presidente** (`CD_CARGO = 1`).

**2. Cálculo do swing em 2022, por geografia**

Para cada candidato, na mesma unidade geográfica (ex.: município):

$$
\\text{share}_{1T} = \\frac{\\text{votos do candidato no 1T}}
{\\text{válidos no 1T}}
\\qquad
\\text{share}_{2T} = \\frac{\\text{votos do candidato no 2T}}
{\\text{válidos no 2T}}
$$

$$
\\text{swing} = \\text{share}_{2T} - \\text{share}_{1T}
$$

Isso é feito individualmente para **Lula** e para **Bolsonaro (2022)**.

**3. Aplicação em 2026**

Mapeamento dos blocos:

| 2022 | 2026 |
|------|------|
| Lula             | Lula             |
| Jair Bolsonaro   | Flávio Bolsonaro |

$$
\\text{share\\_prev}_{2T, 2026} = \\text{share}_{1T, 2026} + \\text{swing}_{2022}
$$

**4. Normalização**

Como a soma dos dois shares pode não dar exatamente 100% (por causa
de terceiros, brancos e nulos no 1T de 2026), reescalamos:

$$
\\text{share\\_norm} = \\frac{\\text{share\\_prev}}
{\\text{share\\_prev}_{Lula} + \\text{share\\_prev}_{Flávio}}
$$

**5. Estimativa de votos absolutos**

O total de votos válidos do 2º turno de 2026 é estimado usando a
**relação observada em 2022** na mesma geografia:

$$
\\text{ratio} = \\frac{\\text{válidos}_{2T, 2022}}
{\\text{válidos}_{1T, 2022}}
$$

$$
\\text{válidos}_{2T, 2026} \\approx
\\text{válidos}_{1T, 2026} \\times \\text{ratio}
$$

$$
\\text{votos\\_prev} = \\text{share\\_norm} \\times \\text{válidos}_{2T, 2026}
$$

---

### Mapa

Cada unidade geográfica é pintada com a cor do **candidato com maior
share projetado**:

- 🔴 **Vermelho** → Lula
- 🔵 **Azul** → Flávio Bolsonaro
- ⚪ **Cinza** → sem dados suficientes para projetar

---

### Limitações

- **Não é uma previsão eleitoral.** É um exercício de transferência
  mecânica do swing observado em 2022.
- Assume que o eleitor de Jair Bolsonaro em 2022 migra integralmente
  para Flávio Bolsonaro em 2026.
- Ignora mudanças de cenário entre 2022 e 2026: alianças estaduais,
  rejeição, abstenção, votos brancos/nulos e entrada/saída de
  eleitores.
- Regiões com **poucos votos** (municípios pequenos) tendem a swing
  ruidoso — interprete com cautela.
- A estimativa de votos absolutos depende da relação
  *válidos 2T / válidos 1T* de 2022, que pode não se repetir em 2026.

---

### Fonte dos dados

Todos os dados são públicos e obtidos diretamente do portal do
**Tribunal Superior Eleitoral (TSE)**:

- **[Votação por seção — 2022 (1º e 2º turnos)](https://cdn.tse.jus.br/estatistica/sead/odsele/votacao_secao/votacao_secao_2022_BR.zip)**
- **[Votação por seção — 2026 (1º turno)](https://cdn.tse.jus.br/estatistica/sead/odsele/votacao_secao/votacao_secao_2026_BR.zip)**
- **[Portal de dados abertos do TSE](https://dadosabertos.tse.jus.br/)**
- **[Malhas territoriais (IBGE / geobr)](https://github.com/ipeaGIT/geobr)**
"""
    )

# ---- KPIs + GRÁFICO ----
st.subheader(f"Resumo — {abrangencia}")

tot_lula   = float(df["v_lula_2T_prev"].sum())
tot_flavio = float(df["v_flavio_2T_prev"].sum())
tot_geral  = tot_lula + tot_flavio or 1.0

c1, c2, c3, c4 = st.columns([1, 1, 1, 1.4])

with c1:
    st.markdown(
        f"<div style='padding:12px;border-left:6px solid {COR_LULA};"
        f"background:#fff5f5;border-radius:6px'>"
        f"<b>Lula</b><br>"
        f"<span style='font-size:22px'>{fmt_int(tot_lula)}</span> votos<br>"
        f"<span style='color:#666'>{tot_lula/tot_geral*100:.2f}% do total</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        f"<div style='padding:12px;border-left:6px solid {COR_FLAVIO};"
        f"background:#f0f6ff;border-radius:6px'>"
        f"<b>Flávio Bolsonaro</b><br>"
        f"<span style='font-size:22px'>{fmt_int(tot_flavio)}</span> votos<br>"
        f"<span style='color:#666'>{tot_flavio/tot_geral*100:.2f}% do total</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

with c3:
    vant = tot_lula - tot_flavio
    quem = "Lula" if vant >= 0 else "Flávio"
    cor  = COR_LULA if vant >= 0 else COR_FLAVIO
    st.markdown(
        f"<div style='padding:12px;border-left:6px solid {cor};"
        f"background:#fafafa;border-radius:6px'>"
        f"<b>Vantagem</b><br>"
        f"<span style='font-size:22px'>{quem}</span><br>"
        f"<span style='color:#666'>{fmt_int(abs(vant))} votos "
        f"({abs(vant)/tot_geral*100:.2f} p.p.)</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

with c4:
    df_bar = pd.DataFrame({
        "Candidato": ["Lula", "Flávio"],
        "Votos":     [tot_lula, tot_flavio],
    })
    fig_bar = px.bar(
        df_bar,
        x="Votos",
        y="Candidato",
        orientation="h",
        text="Votos",
        color="Candidato",
        color_discrete_map={
            "Lula":   COR_LULA,
            "Flávio": COR_FLAVIO,
        },
    )
    fig_bar.update_traces(
        texttemplate="%{x:,.0f}",
        textposition="outside",
        cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>%{x:,.0f} votos<extra></extra>",
    )
    fig_bar.update_layout(
        height=140,
        margin=dict(l=0, r=40, t=10, b=10),
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(title=None, autorange="reversed"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        bargap=0.35,
    )
    st.plotly_chart(
        fig_bar,
        width="stretch",
        config={"displayModeBar": False, "responsive": True},
    )

# ---- MAPA ----
st.subheader("Mapa")

if not tem_geo():
    st.warning(
        "Para ver os mapas, instale as bibliotecas:\n\n"
        "```\npip install geopandas geobr\n```"
    )
else:
    # ---------- Nacional (todos os municípios do Brasil) ----------
    if chave == "nacional":
        try:
            with st.spinner("Cruzando municípios com dados do TSE…"):
                df_mun = adiciona_vencedor(carregar(ARQUIVOS["Municipal"][0]))
                df_mun["nome_norm"] = df_mun["NM_MUNICIPIO"].apply(normaliza)
                df_mun["uf_norm"]   = df_mun["SG_UF"].astype(str).str.upper()

                gdf = geo_municipios_brasil().copy()
                gdf["nome_norm"] = gdf["name_muni"].apply(normaliza)
                gdf["uf_norm"]   = gdf["abbrev_state"].astype(str).str.upper()

                d = gdf.merge(
                    df_mun[["uf_norm", "nome_norm",
                            "share_lula_2T_prev", "share_flavio_2T_prev",
                            "v_lula_2T_prev", "v_flavio_2T_prev",
                            "vencedor"]],
                    on=["uf_norm", "nome_norm"],
                    how="left"
                )
                d["vencedor"] = d["vencedor"].fillna("Sem dados")
                d = d.reset_index(drop=True)
                d["_fid"] = d.index.astype(str)

            fig = mapa_choropleth(
                d, "_fid", "name_muni",
                {
                    "share_lula_2T_prev":   ":.2%",
                    "share_flavio_2T_prev": ":.2%",
                    "v_lula_2T_prev":       ":,.0f",
                    "v_flavio_2T_prev":     ":,.0f",
                },
                "Vencedor projetado — municípios do Brasil",
                height=620,
            )
            st.plotly_chart(
                fig,
                width="stretch",
                config={"displayModeBar": False, "responsive": True},
            )

            sem = (d["vencedor"] == "Sem dados").sum()
            if sem:
                st.caption(
                    f"{sem} municípios do shapefile não foram casados com "
                    "os dados (nome divergente ou sem dados)."
                )
            st.caption(
                "Dica: se ainda estiver lento, escolha **Estadual** no menu "
                "lateral — é bem mais leve."
            )
        except Exception as e:
            st.error(f"Erro ao montar o mapa nacional: {e}")

    # ---------- Estadual ----------
    elif chave == "estado":
        try:
            gdf = geo_estados().copy()
            d = gdf.merge(df, left_on="abbrev_state", right_on="SG_UF", how="left")
            d["vencedor"] = d["vencedor"].fillna("Sem dados")
            d = d.reset_index(drop=True)
            d["_fid"] = d.index.astype(str)

            fig = mapa_choropleth(
                d, "_fid", "abbrev_state",
                {
                    "share_lula_2T_prev":   ":.2%",
                    "share_flavio_2T_prev": ":.2%",
                    "v_lula_2T_prev":       ":,.0f",
                    "v_flavio_2T_prev":     ":,.0f",
                },
                "Vencedor projetado por estado",
            )
            st.plotly_chart(
                fig,
                width="stretch",
                config={"displayModeBar": False, "responsive": True},
            )
        except Exception as e:
            st.error(f"Erro ao montar o mapa estadual: {e}")

    # ---------- Municipal ----------
    else:
        if uf_sel is None:
            st.info("Selecione uma UF no menu lateral para carregar o mapa de municípios.")
        else:
            try:
                gdf = geo_municipios_uf(uf_sel).copy()
                gdf["nome_norm"] = gdf["name_muni"].apply(normaliza)

                d2 = df.copy()
                d2["nome_norm"] = d2["NM_MUNICIPIO"].apply(normaliza)

                d = gdf.merge(
                    d2[["nome_norm",
                        "share_lula_2T_prev", "share_flavio_2T_prev",
                        "v_lula_2T_prev", "v_flavio_2T_prev",
                        "vencedor"]],
                    on="nome_norm",
                    how="left"
                )
                d["vencedor"] = d["vencedor"].fillna("Sem dados")
                d = d.reset_index(drop=True)
                d["_fid"] = d.index.astype(str)

                fig = mapa_choropleth(
                    d, "_fid", "name_muni",
                    {
                        "share_lula_2T_prev":   ":.2%",
                        "share_flavio_2T_prev": ":.2%",
                        "v_lula_2T_prev":       ":,.0f",
                        "v_flavio_2T_prev":     ":,.0f",
                    },
                    f"Vencedor projetado — municípios de {uf_sel}",
                    auto_zoom=True,
                )
                st.plotly_chart(
                    fig,
                    width="stretch",
                    config={"displayModeBar": False, "responsive": True},
                )

                sem = (d["vencedor"] == "Sem dados").sum()
                if sem:
                    st.caption(
                        f"{sem} municípios do shapefile não foram casados "
                        "com os dados."
                    )
            except Exception as e:
                st.error(f"Erro ao montar o mapa de municípios: {e}")

# ---- TABELA ----
st.subheader("Detalhamento")

cols_pref = ["SG_UF", "NM_MUNICIPIO",
             "validos_1T_26",
             "share_lula_2T_prev", "share_flavio_2T_prev",
             "v_lula_2T_prev", "v_flavio_2T_prev",
             "vencedor"]

cols_show = [c for c in cols_pref if c in df.columns]
df_show = df[cols_show].copy()

ordem_col = "v_lula_2T_prev" if "v_lula_2T_prev" in df_show.columns else df_show.columns[0]
df_show = df_show.sort_values(ordem_col, ascending=False)

limite = LIMITE_TABELA.get(chave, 20000)
if len(df_show) > limite:
    st.caption(
        f"Mostrando as {limite:,} linhas com mais votos para Lula "
        f"(de {len(df_show):,}).".replace(",", ".")
    )
    df_show = df_show.head(limite)

col_cfg = {
    "SG_UF":                st.column_config.TextColumn("UF"),
    "NM_MUNICIPIO":         st.column_config.TextColumn("Município"),
    "validos_1T_26":        st.column_config.NumberColumn("Válidos 1T 2026", format="%d"),
    "share_lula_2T_prev":   st.column_config.NumberColumn("Share Lula 2T",   format="%.4f"),
    "share_flavio_2T_prev": st.column_config.NumberColumn("Share Flávio 2T", format="%.4f"),
    "v_lula_2T_prev":       st.column_config.NumberColumn("Votos Lula 2T",   format="%d"),
    "v_flavio_2T_prev":     st.column_config.NumberColumn("Votos Flávio 2T", format="%d"),
    "vencedor":             st.column_config.TextColumn("Vencedor"),
}

st.dataframe(
    df_show,
    column_config=col_cfg,
    width="stretch",
    hide_index=True,
    height=420,
)

# ---- DOWNLOAD ----
st.download_button(
    "⬇️ Baixar CSV desta seleção",
    data=df.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig"),
    file_name=f"previsao_2T_2026_{chave}.csv",
    mime="text/csv",
)

st.caption(
    f"Fonte: dados públicos do TSE · "
    f"Total de linhas nesta seleção: {len(df):,}".replace(",", ".")
)
