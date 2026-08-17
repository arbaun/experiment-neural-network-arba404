import numpy as np
import pickle
model_data = np.load("bobot_w.npz")
w = model_data["bobot"]
b = model_data["bias"]
with open("kamus.pkl",'rb') as f:
     vocab = pickle.load(f)
def generate_target(text, kamus, ukuran_kamus):
    kata_kata = text.split()
    panjang_sekuens = len(kata_kata)
    matrix_target = np.zeros((panjang_sekuens,ukuran_kamus))
    for i, kata in enumerate(kata_kata):
        if kata in kamus:
           indeks_kata = kamus[kata]
           matrix_target[i, indeks_kata] = 1
    return matrix_target
text_p = input("masukkan pertanyaan atau pernyataan yang akan direspon komputer: ")
text_bersih = text_p.lower().replace("?","")
pencegat = {"siapa yang":"siapa_yang","apa bahasa pemrograman":"apa_bahasa_pemrograman",
"favoritmu":"favorit_mu", "apa os yang":"apa_os_yang",
"kamu pakai":"kamu_pakai", "kamu main":"kamu_main", "game apa":"game_apa",
"hewan apa yang":"hewan_apa_yang", "kamu suka":"kamu_suka",
"jam berapa sekarang":"jam berapa_sekarang",
"bagaimana cara install":"bagaimana_cara_install",
"apakah debian":"apakah_debian","siapa penemu":"siapa_penemu",
"lampu bohlam":"lampu_bohlam","siapa kamu":"siapa_kamu",
"mengapa pakai":"mengapa_pakai","kenapa kucing punya":"kenapa_kucing_punya"
,"sekarang jam berapa":"sekarang jam_berapa",
"apa kelebihan":"apa_kelebihan","pisang atau anggur":"pisang_atau_anggur",
"apa yang sedang kamu kerjakan":"apa_yang_sedang kamu_kerjakan",
"hari ini tanggal berapa":"hari_ini_tanggal_berapa"}
for kata_asli, kata_gabung in pencegat.items():
    text_bersih = text_bersih.replace(kata_asli, kata_gabung)
x_per = generate_target(text_bersih, vocab, len(vocab))
print(x_per)
prob_pre = np.dot(x_per,w)
#print(prob_pre)
index_resp = np.argmax(prob_pre, axis=-1)
#print(index_resp)
kata_kata =[]
for v in index_resp:
    for k,v1 in vocab.items():
        if v1== v:
           kata_kata.append(k)

print(" ".join(kata_kata).replace("_"," "))
