import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Import the Dataset
df = pd.read_csv("skincare.csv")

# Sidebar Dropdown
st.sidebar.title("✨Rekomendasi Produk Skincare✨")
page = st.sidebar.selectbox(
    "Pilih Halaman",
    ["Home", "Dataset", "Visualisasi", "Sistem Rekomendasi", "Tentang Pembuat"]
)



# Home Page
if page == "Home":
    st.title("🌟Sistem Rekomendasi Skincare Berdasarkan Jenis Kulit🌟")
    st.image("skincare.jpeg",use_container_width=True)
    st.markdown("""
    ### Deskripsi
    Website ini bertujuan memberikan rekomendasi skincare yang cocok digunakan berdasarkan kondisi jenis kulit.
    """)

# Dataset Page
elif page == "Dataset":
    st.title("🌟Sistem Rekomendasi Skincare Berdasarkan Jenis Kulit🌟")
    # Deskripsi Dataset
    st.markdown("""
    ### Tentang Dataset
    Dataset ini berisi informasi tentang berbagai produk skincare. Dataset terdiri dari beberapa kolom penting seperti:
    - **product_name**: Nama produk.
    - **product_type**: Jenis produk (Facial wash, Toner, Serum, Moisturizer, Sunscreen).
    - **brand**: Merek produk.
    - **price**: Harga produk dalam IDR.
    - **skin_type**: Jenis kulit yang sesuai (Normal, Dry, Oily, Combination, Sensitive).
    - **notable_effects**: Manfaat utama dari produk.
    - **description**: Deskripsi singkat produk.
    """)
    st.write("### Dataset:")
    st.write(df.head(10))  # Tampilkan 10 baris pertama dari dataset


# Visualisasi Page
elif page == "Visualisasi":
    st.title("🌟Sistem Rekomendasi Skincare Berdasarkan Jenis Kulit🌟")
    st.write("### 10 TOP Brand Skincare")
    # Brand
    # visualisasi 1
    counts_brand = df['brand'].value_counts()
    count_percentage = df['brand'].value_counts(1) * 100
    counts_dfbrand = pd.DataFrame(
        {'Brand': counts_brand.index, 'Counts': counts_brand.values, 'Percent%': np.round(count_percentage.values, 2)})
    top_10_brands = counts_dfbrand.head(10)

    plt.figure(figsize=(10, 5))
    sns.set(style='white')
    ax = sns.barplot(x='Brand', y='Counts', width=0.6, data=top_10_brands, palette='magma')
    ax.set_title('Total Products of Top 10 Brands', fontsize=15, fontweight='bold')
    ax.set_xlabel('Brand', fontsize=12, fontweight='medium')
    ax.set_ylabel('Total Products', fontsize=12, fontweight='medium')

    for label in ax.containers:
        ax.bar_label(label, fontweight='medium', fontsize=10)
    plt.xticks(rotation=15, fontsize=10)
    st.pyplot(plt)

    # visualisasi 2
    st.write("### Tipe Produk untuk Basic Skincare")
    product_counts = df['product_type'].value_counts()
    plt.figure(figsize=(6, 6))
    sns.set(style='white')

    # Creating a pie chart first
    sizes = product_counts
    labels = product_counts.index
    colors = sns.color_palette("Set2", len(labels))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=70, wedgeprops=dict(width=0.4))
    plt.title('Basic Skincare', fontsize=15, fontweight='bold')
    plt.ylabel('')
    st.pyplot(plt)

    # visualisasi 3
    st.write("### Jenis Permasalahan Kulit")
    plt.figure(figsize=(8, 5))
    sns.countplot(x='skintype', data=df, palette='Set2', )
    plt.title('Distribusi Produk Berdasarkan Jenis Kulit')
    plt.xlabel('Skin type')
    plt.ylabel('Total product')
    plt.xticks(rotation=45, ha='right')
    st.pyplot(plt)

elif page == "Sistem Rekomendasi":
    st.title("🌟Sistem Rekomendasi Skincare Berdasarkan Jenis Kulit🌟")
    st.write(
        """
        ##### **Untuk mendapatkan rekomendasi, silahkan masukkan jenis produk yang diinginkan, jenis kulit anda, permasalahan kulit anda serta dampak yang anda inginkan**
        """)

    st.write('---')

    first, last = st.columns(2)

    category = first.selectbox(label='Kategori Produk: ', options=df['product_type'].unique())
    category_pt = df[df['product_type'] == category]

    # Choose a skin type
    # st = skin type
    skin_type = last.selectbox(label='Pilih Jenis Kulit : ',
                               options=['Normal', 'Dry', 'Oily', 'Combination', 'Sensitive'])
    category_st_pt = category_pt[category_pt[skin_type] == 1]

    # pilih keluhan
    prob = st.multiselect(label='Permasalahan Kulit : ',
                          options=['Kulit Kusam', 'Jerawat', 'Bekas Jerawat', 'Pori-pori Besar', 'Flek Hitam',
                                   'Garis Halus dan Kerutan', 'Komedo', 'Warna Kulit Tidak Merata', 'Kemerahan',
                                   'Kulit Kendur'])

    # Choose notable_effects
    # dari produk yg sudah di filter berdasarkan product type dan skin type(category_st_pt), kita akan ambil nilai yang unik di kolom notable_effects
    opsi_ne = category_st_pt['notable_effects'].unique().tolist()
    # notable_effects-notable_effects yang unik maka dimasukkan ke dalam variabel opsi_ne dan digunakan untuk value dalam multiselect yg dibungkus variabel selected_options di bawah ini
    selected_options = st.multiselect('Dampak Yang Diinginkan : ', opsi_ne)
    # hasil dari selected_options kita masukan ke dalam var category_ne_st_pt
    category_ne_st_pt = category_st_pt[category_st_pt["notable_effects"].isin(selected_options)]

    # Choose product
    # produk2 yang sudah di filter dan ada di var filtered_df kemudian kita saring dan ambil yang unik2 saja berdasarkan product_name dan di masukkan ke var opsi_pn
    opsi_pn = category_ne_st_pt['product_name'].unique().tolist()
    # buat sebuah selectbox yang berisi pilihan produk yg sudah di filter di atas
    product = st.selectbox(label='Produk yang Direkomendasikan Untuk Kamu', options=sorted(opsi_pn))
    # variabel product di atas akan menampung sebuah produk yang akan memunculkan rekomendasi produk lain

    ## MODELLING with Content Based Filtering
    # Inisialisasi TfidfVectorizer
    tf = TfidfVectorizer()

    # Melakukan perhitungan idf pada data 'notable_effects'
    tf.fit(df['notable_effects'])

    # Mapping array dari fitur index integer ke fitur nama
    tf.get_feature_names_out()

    # Melakukan fit lalu ditransformasikan ke bentuk matrix
    tfidf_matrix = tf.fit_transform(df['notable_effects'])

    # Melihat ukuran matrix tfidf
    shape = tfidf_matrix.shape

    # Mengubah vektor tf-idf dalam bentuk matriks dengan fungsi todense()
    tfidf_matrix.todense()

    # Membuat dataframe untuk melihat tf-idf matrix
    # Kolom diisi dengan efek-efek yang diinginkan
    # Baris diisi dengan nama produk
    pd.DataFrame(
        tfidf_matrix.todense(),
        columns=tf.get_feature_names_out(),
        index=df.product_name
    ).sample(shape[1], axis=1).sample(10, axis=0)

    # Menghitung cosine similarity pada matrix tf-idf
    cosine_sim = cosine_similarity(tfidf_matrix)

    # Membuat dataframe dari variabel cosine_sim dengan baris dan kolom berupa nama produk
    cosine_sim_df = pd.DataFrame(cosine_sim, index=df['product_name'], columns=df['product_name'])

    # Melihat similarity matrix pada setiap nama produk
    cosine_sim_df.sample(7, axis=1).sample(10, axis=0)


    # Membuat fungsi untuk mendapatkan rekomendasi
    def skincare_recommendations(nama_produk, similarity_data=cosine_sim_df,
                                 items=df[['product_name', 'brand', 'description']], k=5):
        # Mengambil data dengan menggunakan argpartition untuk melakukan partisi secara tidak langsung sepanjang sumbu yang diberikan
        # Dataframe diubah menjadi numpy
        # Range(start, stop, step)
        index = similarity_data.loc[:, nama_produk].to_numpy().argpartition(range(-1, -k, -1))

        # Mengambil data dengan similarity terbesar dari index yang ada
        closest = similarity_data.columns[index[-1:-(k + 2):-1]]

        # Drop nama_produk agar nama produk yang dicari tidak muncul dalam daftar rekomendasi
        closest = closest.drop(nama_produk, errors='ignore')
        df = pd.DataFrame(closest).merge(items).head(k)
        return df


    # Membuat button untuk menampilkan rekomendasi
    model_run = st.button('Temukan Rekomendasi Produk Lainnya!')
    # Mendapatkan rekomendasi
    if model_run:
        st.write('Berikut Rekomendasi Produk Serupa Lainnya Sesuai yang Kamu Inginkan')
        st.write(skincare_recommendations(product))

# Tentang Pembuat Page
elif page == "Tentang Pembuat":
    st.title("🌺Tentang Pembuat🌺")
    st.markdown("""
    - **Nama**: [Annisa Qurrotaa'yun]
    - **NIM**: [223307034]
    - **Kelas**: [5B]
    - **Proyek Dibuat Untuk**: [Tugas Mata Kuliah Data Science]
    - **Sumber Kode**: [GitHub Link]
    """)





