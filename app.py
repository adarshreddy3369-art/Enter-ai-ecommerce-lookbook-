import os
import json 
import re
import numpy as np
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
from openai import OpenAI




# =========================================================
# LOAD API KEY
# =========================================================                                                 

api_key = os.getenv("OPENAI_API_KEY", "").strip()

# A valid OpenAI key starts with sk-. Do not let a placeholder or an expired
# key take down the entire catalog: semantic search has a local fallback.
client = OpenAI(api_key=api_key) if api_key.startswith("sk-") else None
key_configuration_error = bool(api_key) and client is None


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI E-Commerce Lookbook",
    page_icon="🛍️",
    layout="wide"
)


# =========================================================
# LOAD PRODUCTS
# =========================================================

@st.cache_data
def load_products():

    with open("products.json", "r", encoding="utf-8") as file:
        data = json.load(file)

    return pd.DataFrame(data)


df = load_products()


# =========================================================
# PRODUCT TEXT FOR AI
# =========================================================

def create_product_text(product):

    return f"""
    Product Name: {product['name']}
    Category: {product['category']}
    Tags: {', '.join(product['tags'])}
    Description: {product['description']}
    """


# =========================================================
# CREATE EMBEDDING
# =========================================================

def local_embedding(text, dimensions=384):
    """Small credential-free fallback vector for graceful semantic discovery."""
    vector = np.zeros(dimensions, dtype=float)
    for token in re.findall(r"[a-z]+", text.lower()):
        vector[hash(token) % dimensions] += 1
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


# =========================================================
# SEMANTIC SEARCH
# =========================================================

def semantic_search(query, products_df):
    product_texts = [create_product_text(product) for _, product in products_df.iterrows()]

    # Request all vectors in one call. Authentication and network failures are
    # handled here, so a bad key never crashes the Streamlit page.
    if client:
        try:
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=[query, *product_texts]
            )
            vectors = np.array([item.embedding for item in response.data], dtype=float)
        except Exception:
            st.session_state["embedding_notice"] = (
                "OpenAI embeddings are unavailable, so local semantic matching is being used. "
                "Check your API key and account access to re-enable hosted AI."
            )
            vectors = np.vstack([local_embedding(text) for text in [query, *product_texts]])
    else:
        vectors = np.vstack([local_embedding(text) for text in [query, *product_texts]])

    query_embedding, product_embeddings = vectors[0], vectors[1:]
    denominator = np.linalg.norm(product_embeddings, axis=1) * np.linalg.norm(query_embedding) + 1e-9
    scores = (product_embeddings @ query_embedding / denominator).tolist()

    # Copy product DataFrame
    result_df = products_df.copy()

    # Add similarity scores
    result_df["similarity"] = scores

    # Sort highest similarity first
    result_df = result_df.sort_values(
        by="similarity",
        ascending=False
    )

    return result_df


# =========================================================
# TITLE
# =========================================================

st.title("🛍️ AI E-Commerce Lookbook")

if key_configuration_error:
    st.warning(
        "Your OPENAI_API_KEY looks like a placeholder, not a valid OpenAI key. "
        "AI search will use local matching until you replace it with a key beginning with `sk-`."
    )
elif not api_key:
    st.info("No OpenAI key found. AI search is using local matching.")

st.write(
    "Explore products using normal filters or describe what you are looking for using AI."
)


# =========================================================
# AI SEMANTIC SEARCH
# =========================================================

st.subheader("🤖 AI Product Search")

ai_query = st.text_input(
    "Describe what you are looking for",
    placeholder="Example: something comfortable for exercise"
)


# =========================================================
# SIDEBAR FILTERS
# =========================================================

st.sidebar.header("🔎 Product Filters")


# Normal search
search = st.sidebar.text_input(
    "Search products",
    placeholder="Search by name or description..."
)


# Category
categories = ["All"] + sorted(
    df["category"].unique().tolist()
)

selected_category = st.sidebar.selectbox(
    "Category",
    categories
)


# Price
min_price = int(df["price"].min())
max_price = int(df["price"].max())

price_range = st.sidebar.slider(
    "Price Range",
    min_value=min_price,
    max_value=max_price,
    value=(min_price, max_price),
    step=100
)


# Tags
all_tags = sorted(
    set(
        tag
        for tags in df["tags"]
        for tag in tags
    )
)

selected_tags = st.sidebar.multiselect(
    "Tags",
    all_tags
)


# =========================================================
# NORMAL FILTERING
# =========================================================

filtered_df = df.copy()


# Search filter
if search:

    search_text = search.lower()

    filtered_df = filtered_df[
        filtered_df["name"]
        .str.lower()
        .str.contains(search_text, na=False)
        |
        filtered_df["description"]
        .str.lower()
        .str.contains(search_text, na=False)
    ]


# Category filter
if selected_category != "All":

    filtered_df = filtered_df[
        filtered_df["category"] == selected_category
    ]


# Price filter
filtered_df = filtered_df[
    (filtered_df["price"] >= price_range[0])
    &
    (filtered_df["price"] <= price_range[1])
]


# Tag filter
if selected_tags:

    filtered_df = filtered_df[
        filtered_df["tags"].apply(
            lambda product_tags:
            all(tag in product_tags for tag in selected_tags)
        )
    ]


# =========================================================
# AI SEARCH RESULTS
# =========================================================

if ai_query:

    st.subheader("✨ AI Recommended Products")

    with st.spinner("AI is finding the most relevant products..."):

        ai_results = semantic_search(
            ai_query,
            df
        )

    if notice := st.session_state.pop("embedding_notice", None):
        st.warning(notice)

    ai_results = ai_results.head(3)

    st.write(
        f"Results for: **{ai_query}**"
    )

    for _, product in ai_results.iterrows():

        similarity_percentage = round(
            product["similarity"] * 100,
            1
        )

        st.write(
            f"**{product['name']}** — "
            f"{similarity_percentage}% semantic match"
        )


# =========================================================
# PRODUCT DETAILS MODAL
# =========================================================

@st.dialog("Product Details")
def show_product_details(product):

    st.image(
        product["image_url"],
        width="stretch"
    )

    st.subheader(product["name"])

    st.write(
        f"**Category:** {product['category']}"
    )

    st.write(
        f"**Price:** ₹{product['price']:,}"
    )

    st.write(
        f"**Description:** {product['description']}"
    )

    st.write(
        "**Tags:** " + ", ".join(product["tags"])
    )

    st.divider()

    stock = product["stock"]

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Stock Available",
            stock
        )

    with col2:

        if stock > 20:
            status = "In Stock"

        elif stock > 0:
            status = "Limited Stock"

        else:
            status = "Out of Stock"

        st.metric(
            "Availability",
            status
        )

    max_stock = 40

    stock_percentage = min(
        stock / max_stock,
        1.0
    )

    st.progress(
        stock_percentage
    )


# =========================================================
# RESULT COUNT
# =========================================================

st.subheader(
    f"Products Found: {len(filtered_df)}"
)


# =========================================================
# PRODUCT GRID
# =========================================================

if filtered_df.empty:

    st.warning(
        "No products found. Try changing your filters."
    )

else:

    products = filtered_df.to_dict("records")

    for i in range(0, len(products), 3):

        columns = st.columns(3)

        for j, col in enumerate(columns):

            index = i + j

            if index >= len(products):
                break

            product = products[index]

            with col:

                with st.container(border=True):

                    st.image(
                        product["image_url"],
                        width="stretch"
                    )

                    st.subheader(
                        product["name"]
                    )

                    st.caption(
                        f"🏷️ {product['category']}"
                    )

                    st.markdown(
                        f"### ₹{product['price']:,}"
                    )

                    description = product["description"]

                    if len(description) > 100:

                        description = (
                            description[:100] + "..."
                        )

                    st.write(
                        description
                    )

                    st.caption(
                        f"📦 Stock: {product['stock']}"
                    )

                    if st.button(
                        "View Details",
                        key=f"details_{product['id']}",
                        width="stretch"
                    ):

                        show_product_details(product)
