from collections import defaultdict
import numpy as np
import pickle
import matplotlib.pyplot as plt
#corpus embed buat test
corpus = ["saya suka Fedora GNU/Linux", "kamu benci install freebsd","saya membeli laptop baru","Hari ini cuaca cerah","Hari ini hujan"
,"dia membeli buku","python bahasa pemrograman keren",
"Gnome 51 beta enak juga ya","fedora rawhide update tersedia setiap hari gnu/linux","astaga korpusku hancur",
"aku_sendang_berusaha membuat_text_generatif text",
"inilah efek dari eksperimen","apapun hasilnya aku bahagia",
"yang penting aku bisa membuat kalimat baru favorit",
"game yang aku mainin cuma efootball","sedia payung sebelum hujan",
"baru kali ini aku menikmati ngoding convolutional neural network",
"kucing itu lucu banget","komputerku isinya opensource doang","besok akan turun hujan deras",
"lusa akan lebih baik lagi","os debian seru loh","pakai apa saja tetep menarik"
,"Linus Torvalds pembuat kernel linux","harus ada kernel_linux tentu saja_python favoritmu siapa",
"kau kowe karena apa hewan main ya apakah pernah terjadi berapa jam sekarang lima dong favorit_saya apa_os_yang apa_bahasa_pemrograman kamu_suka berapa_sekarang",
"siapa_yang tentu_saja python_favorit_saya kamu_main game_apa ",
"favorit_mu apa_os_yang kamu_pakai aku_pakai aku_main berapa_sekarang jam_lima",
"hewan_apa_yang kamu_suka aku_suka jam_aksi",
"bagaimana_cara_install distro tutorial_install","apakah_debian cocok buat pemula iya,_debian , bagaimana kenapa berapa siapa_penemu lampu_bohlam Thomas_Alpha_Edison yang_membuat_lampu_bohlam . ",
"siapa_kamu bot aku_jerukzerobot_sebuah_bot_ai_sederhana_yang_diciptakan_sebagai_experimen_kecil-kecilan_maaf_ya_kalau_terbatas_kemampuannya",
"mengapa_pakai categorical cross entropy karena_secara desain lebik_cocok buat_generatif_text",
"kenapa_kucing_punya kumis karena_untuk mengukur_jarak_suatu_celah_supaya_pas_sama_tubuhnya",
"jam_berapa tanya_waktu anggur pisang mangga ","apa_yang_sedang kamu_kerjakan",
"pisang_atau_anggur apa_kelebihan linuxmint lebih_mudah bagi_pemula hari_ini_tanggal_berapa tanya_tanggal",
"apa_kekurangan banyak, salah_satunya_GIL_Global_Interpreter_Lock. namun_fitur_dictionary-nya membuat_sulit_meninggalkan-nya",
"hari_ini_hari_apa tanya_hari kapan_kamu setiap_aku_bosan pasti_main_game main_game bosan pasti pepatah_apa yang ingat",
"fungsi apa_fungsi",
"software yang_menghubungkan_sistem_operasi_ke_hardware_yang_ada_di_komputer_dan_juga_peripheralnya_tapi,_harus_ada_kernel_modul_untuk_mengenali_perangkat."
]
#kamus vocab pakai defaultdict
vocab = defaultdict(lambda: len(vocab))
kernel = np.array([[1,0,-1,0],
[0,1,0,-1],
[1,1,0,0]])
tokenized_corpus = []
for sentence in corpus:
    tokens = sentence.split()
    token_ids = [vocab[token] for token in tokens]
    tokenized_corpus.append(token_ids)

total_vocab_size = len(vocab)
#print("vocabulary kamus: ", dict(vocab))
#print("token_ids: ",tokenized_corpus)
#np.save("bobot_w.npy", w)
#w= np.load("bobot_w.npy")
#import pickle
#with open("kamus.pkl",'wb') as f:
#     pickle.dump(dict(vocab),f)
#with open("kamus.pkl",'rb') as f:
#     vocab = pickle.load(f)
EMBEDDING_DIM =4
np.random.seed(total_vocab_size)
embedding_matrix = np.random.randn(total_vocab_size,EMBEDDING_DIM)
def text_to_matrix(sentence, vocab, embedding_matrix, max_len=total_vocab_size):
    tokens= sentence.split()
    matrix = []
    for token in tokens:
        word_id = vocab.get(token, 0)
        vector = embedding_matrix[word_id]
        matrix.append(vector)
    while len(matrix)< max_len:
        matrix.append(np.zeros(EMBEDDING_DIM))
    matrix = matrix[:max_len]
    return np.array(matrix)
def text_conv1d(matrix, filt):
    seq_len, embedding_dim = matrix.shape
    k_size, _ = filt.shape
    out_len = seq_len - k_size+1
    output = np.zeros(out_len)
    for i in range(out_len):
        window = matrix[i:i+k_size:]
        output[i] = np.sum(window * filt)
    return output
def softmax(x):
    exp_x = np.exp(x - np.max(x,axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x,axis=-1, keepdims=True)
def forward(x,w,b):
    z = np.dot(x,w)
    out = softmax(z)
    return softmax(z), out
def compute_loss(y_pred, y_true):
    return -np.mean(np.sum(y_true * np.log(y_pred + 1e-9),axis=-1))
def backward(x, y_true, y_pred,z, w, b, learning_rate=0.01):
    grad_loss = (y_pred - y_true)/x.shape[0]
    grad_activation = grad_loss *(z>0)
    grad_w = np.dot(x.T, grad_activation)
    grad_b = np.sum(grad_activation)
    w_updated = w -(learning_rate * grad_w)
    b_updated = b -(learning_rate * grad_b)
    return w_updated, b_updated
def generate_target(text, kamus, ukuran_kamus):
    kata_kata = text.split()
    panjang_sekuens = len(kata_kata)
    matrix_target = np.zeros((panjang_sekuens,ukuran_kamus))
    for i, kata in enumerate(kata_kata):
        if kata in kamus:
           indeks_kata = kamus[kata]
           matrix_target[i, indeks_kata] = 1
    return matrix_target
w = np.random.randn(total_vocab_size,total_vocab_size) * np.sqrt(2.0/total_vocab_size)
b = np.zeros(total_vocab_size)
#dummy_data
x_train = np.array([[1,0,1,0],[0,1,0,1]])
y_target = np.array([[0,1,0,1],[1,0,1,0]])
#print(total_vocab_size)
#for epoch in range(100):
#    z, y_pred = forward(x_train, w)
#    loss = compute_loss(y_pred, y_target)
#    w = backward(x_train, y_target, y_pred, z, w, learning_rate=0.01)
#    if epoch % 20 == 0:
#       print(f"Epoch {epoch}, Loss: {loss:.4f}")
#       print(y_pred)
#       print(y_target)
kalimat_baru = "lusa akan turun hujan"
print(f"kata manusia: '{kalimat_baru}'")
ktn ="siapa_yang membuat kernel linux"
ktn2 = "apa_bahasa_pemrograman favorit_mu"
j3wb = "Linus Torvalds membuat kernel_linux"
j4wb = "tentu_saja python_favorit_saya"
#tokens_k_baru = kalimat_baru.split()
sentence_matrix = text_to_matrix(kalimat_baru, vocab, embedding_matrix, max_len=total_vocab_size)
conv_out = text_conv1d(sentence_matrix, kernel)
#print(conv_out)
pooled_out = np.max(conv_out,axis=-1)
#print(conv_out)
text_expec = "Hari ini hujan deras"
#expec_matrix = text_to_matrix(text_expec, vocab, embedding_matrix, max_len=total_vocab_size)
#conv_ex = text_conv1d(expec_matrix, kernel)
p_tanya = "apa_os_yang kamu_pakai"
print(f"kata manusia: {p_tanya}?")
te_jawaban = "aku_pakai Fedora"
data_pj = [(kalimat_baru,text_expec),(p_tanya,te_jawaban),
(ktn,j3wb),(ktn2,j4wb),("kamu_main game_apa","aku_main efootball"),("hewan_apa_yang kamu_suka","aku_suka kucing"),
("jam berapa_sekarang","sekarang jam_aksi"),
("bagaimana_cara_install distro gnu/linux","tutorial_install distro gnu/linux"),
("apakah_debian cocok buat pemula","iya,_debian cocok buat pemula"),
("siapa_penemu lampu_bohlam","Thomas_Alpha_Edison yang_membuat_lampu_bohlam"),
("siapa_kamu","aku_jerukzerobot_sebuah_bot_ai_sederhana_yang_diciptakan_sebagai_experimen_kecil-kecilan_maaf_ya_kalau_terbatas_kemampuannya"),
("mengapa_pakai categorical cross entropy", "karena_secara desain lebik_cocok buat_generatif_text"),
("kenapa_kucing_punya kumis","karena_untuk mengukur_jarak_suatu_celah_supaya_pas_sama_tubuhnya"),
("sekarang jam_berapa","sekarang tanya_waktu"),("apa_kelebihan linuxmint","lebih_mudah bagi_pemula"),
("pisang_atau_anggur","anggur"),("apa_yang_sedang kamu_kerjakan","aku_sendang_berusaha membuat_text_generatif"),
("hari_ini_tanggal_berapa","tanya_tanggal"),
("apa_kekurangan bahasa pemrograman python",
"banyak, salah_satunya_GIL_Global_Interpreter_Lock. namun_fitur_dictionary-nya membuat_sulit_meninggalkan-nya"),
("hari_ini_hari_apa","tanya_hari"),("kapan_kamu main_game","setiap_aku_bosan pasti_main_game "),
("pepatah_apa yang kamu ingat","sedia payung sebelum hujan"),
("apa_fungsi kernel","software yang_menghubungkan_sistem_operasi_ke_hardware_yang_ada_di_komputer_dan_juga_peripheralnya_tapi,_harus_ada_kernel_modul_untuk_mengenali_perangkat.")
]
#pencegat = {"siapa yang":"siapa_yang","apa bahasa pemrograman":"apa_bahasa_pemrograman",
#"favoritmu":"favorit_mu", "apa os yang":"apa_os_yang",
#"kamu pakai":"kamu_pakai", "kamu main":"kamu_main", "game apa":"game_apa",
#"hewan apa yang":"hewan_apa_yang", "kamu suka":"kamu_suka,
#"berapa sekarang":"berapa_sekarang"}
#print(w)
#print(y_target)
data_loss = []
for epoch in range(10000):
    #np.random.shuffle(data_pj)
    for tanya, response in data_pj:
        ex_train= generate_target(tanya, dict(vocab), total_vocab_size)
        y_true = generate_target(response, dict(vocab), total_vocab_size)
        z, y_pred = forward(ex_train, w,b)
        loss = compute_loss(y_pred, y_true)
        w,b = backward(ex_train, y_true, y_pred, z, w, b, learning_rate=0.01)
        data_loss.append(loss)

    #if epoch % 20 == 0:
       #print(f"Epoch {epoch}, Loss: {loss:.4f}")
     #  if epoch==980:
     #     prediksi_probs = y_pred
np.savez("bobot_w.npz", bobot=w,bias=b)
with open("kamus.pkl",'wb') as f:
     pickle.dump(dict(vocab),f)
plt.figure(figsize=(8,4))
plt.plot(data_loss, label="training loss", color="orangered",linewidth=2)
plt.title("kurva loss",fontsize=12,fontweight="bold")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.grid(True, linestyle='--',alpha=0.6)
plt.legend()
plt.show()
rock = np.dot(ex_train,w)
#print(rock)
probs = softmax(rock)
#print(probs)
kata_terpilih = np.argmax(probs, axis=-1)
#kata_tp = np.argmax(kata_terpilih,axis=0)
#print("index kata yang di prediksi muncul berikutnya:",kata_terpilih)
kata_kata_prediksi = []
for v in kata_terpilih:
    for k,v1 in dict(vocab).items():
        if v1==v:
           kata_kata_prediksi.append(k)
print("kata-kata mesin: ")
print(" ".join(kata_kata_prediksi))
#def hitung_time():
#    sekarang = datetime.datetime.now()
#    return sekarang.strftime("sekarang jam: %Y-%m-%d %H:%M:S") 
#router_aks = {"sekarang jam_aksi":hitung_time}
#hasil_prediksi = " ".join(kata_kata_prediksi)
#if hasil_prediksi in reuter_aks:
#   balasan = reuter_aks[hasil_prediksi]()
#   send_message(balasan.replace(" ","%20"), chat)
#else:
#   send_message(hasil_prediksi.replace(" ","%20"),chat)
