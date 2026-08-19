import json
import urllib.request as internet
import time
from datetime import date
import os
import subprocess as terminal
import numpy as np
import pickle
import datetime
model_tr= np.load("bobot_w.npz")
w = model_tr["bobot"]
b = model_tr["bias"]
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
pencegat = {"siapa yang":"siapa_yang","apa bahasa pemrograman":"apa_bahasa_pemrograman",
"favoritmu":"favorit_mu", "apa os yang":"apa_os_yang",
"kamu pakai":"kamu_pakai", "kamu main game apa":"kamu_main game_apa",
"hewan apa yang":"hewan_apa_yang", "kamu suka":"kamu_suka",
"jam berapa sekarang":"jam berapa_sekarang",
"bagaimana cara install":"bagaimana_cara_install",
"apakah debian":"apakah_debian","siapa penemu":"siapa_penemu",
"lampu bohlam":"lampu_bohlam","siapa kamu":"siapa_kamu",
"mengapa pakai":"mengapa_pakai","kenapa kucing punya":"kenapa_kucing_punya"
,"sekarang jam berapa":"sekarang jam_berapa",
"apa kelebihan":"apa_kelebihan","pisang atau anggur":"pisang_atau_anggur",
"apa yang sedang kamu kerjakan":"apa_yang_sedang kamu_kerjakan",
"hari ini tanggal berapa":"hari_ini_tanggal_berapa",
"apa kekurangan":"apa_kekurangan", "hari ini hari apa":"hari_ini_hari_apa",
"pepatah apa":"pepatah_apa", "kapan kamu main game":"kapan_kamu main_game",
"apa fungsi kernel":"apa_fungsi kernel"}

TOKEN="YOUR_TOKEN_HERE"
URL = "https://api.telegram.org/bot{}/".format(TOKEN)
def get_url(url):
   response = internet.urlopen(url)
   content = response.read().decode('utf-8')
   #print(content)
   return content

def get_json_from_url(url):
   content = get_url(url)
   js = json.loads(content)
   return js

def hitung_time():
    sekarang = datetime.datetime.now()
    return sekarang.strftime("sekarang jam: %H:%M:%S")

def hitung_tanggal():
    sekarang = datetime.datetime.now()
    return sekarang.strftime("hari ini tanggal: %d-%m-%Y")

def cari_hari():
    sekarang = datetime.datetime.now()
    return sekarang.strftime("hari ini hari %A")

def tutorial_install_distro():
    return "Untuk install distro gnu/linux%0A 1.kamu harus mendownload file iso distro dari situs resmi distro. %0A 2.cek checksumnya valid atau tidak. %0A 3. buat bootable pakai rufus,balena etcher dll kalo fedora pakai fedora image writer. %0A 4. jalankan bootable dan ikutin petunjuk dari distronya."

def get_updates(offset=None):
   url = URL+"getUpdates"
   if offset :
      url += "?offset={}".format(offset)
   js = get_json_from_url(url)
   return js

def get_last_update_id(updates):
    update_ids = []
    for update in updates["result"]:
        update_ids.append(int(update["update_id"]))
    return max(update_ids)


def send_message(text, chat_id):
   url = URL+"sendMessage?text={}&chat_id={}".format(text, chat_id)
   get_url(url)

def create_file(text):
   f = open('idbot.txt', 'a')
   f.write(str(text))
   f.close()

def read_file():
   f = open('idbot.txt', 'r+')
   text=f.read()
   f.close()
   return text

def rewrite_file(text):
   f = open('idbot.txt', 'r+')
   f.seek(0)
   f.truncate()
   f.write(str(text))
   f.close()

def tanggap(updates):
   for update in updates["result"]:
      text = update["message"]["text"]
      chat = update["message"]["chat"]["id"]
      if "halo" in text :
          send_message("halo%20juga%20kak",chat)
      #elif text.startswith('aku') :
      #    msg = os.uname()[1]+" "+os.uname()[2]+" "+os.uname()[4]
      #    send_message(msg,chat)
      #elif "boot" in text :
      #    p = terminal.Popen(['systemd-analyze'],stdout=terminal.PIPE,stderr=terminal.PIPE,stdin=terminal.PIPE)
      #    msg,err = p.communicate()
      #    send_message(msg,chat)
      #elif "shutdown" in text:
      #    send_message("shutting down now...",chat)
      #    text_split = text.split(' ',1)
      #    passwd = text_split[1]
      #    command = 'poweroff'
      #    p = terminal.Popen(['sudo', '-S', command], stdin=terminal.PIPE, stderr=terminal.PIPE, universal_newlines=True)
      #    sudo_prompt = p.communicate(passwd + '\n')[1]
      #elif "benchmark" in text:
      #    command = 'sysbench --num-threads=4 --test=cpu --cpu-max-prime=20000 --validate run'.split()
      #    p = terminal.Popen(command,stdout=terminal.PIPE, stderr=terminal.PIPE, stdin=terminal.PIPE)
      #    msg,error = p.communicate()
      #    send_message(msg,chat)
      elif text.startswith('echo'):
          yang_dikirim = text.split(' ',1)[1]
          send_message(yang_dikirim.replace(" ","%20"), chat)
      else:
          text_bersih = text.lower().replace("?","").replace("!","").replace(",","").replace(".","")
          for kata_asli, kata_gabung in pencegat.items():
              text_bersih = text_bersih.replace(kata_asli, kata_gabung)
          x_per = generate_target(text_bersih, vocab, len(vocab))
          prob_pre = np.dot(x_per,w)+b
          index_resp = np.argmax(prob_pre, axis=-1)
          kata_kata =[]
          for v in index_resp:
              for k,v1 in vocab.items():
                  if v1== v:
                      kata_kata.append(k)
          router_aks = {"sekarang jam_aksi":hitung_time,
          "tutorial_install distro gnu/linux":tutorial_install_distro,
          "sekarang tanya_waktu":hitung_time,
           "tanya_tanggal":hitung_tanggal,
           "tanya_hari":cari_hari}
          hasil_prediksi = " ".join(kata_kata)
          if hasil_prediksi in router_aks:
             balasan = router_aks[hasil_prediksi]()
             send_message(balasan.replace(" ","%20"), chat)
          else:
             send_message(hasil_prediksi.replace("_"," ").replace(" ","%20"),chat)


def main():
    last_update_id = None
    while True:
        if os.path.isfile('idbot.txt'):
           last_update_id = int(read_file())
        updates = get_updates(last_update_id)
        if len(updates["result"]) > 0:
           last_update_id = get_last_update_id(updates) + 1
           if not os.path.isfile('idbot.txt'):
              create_file(last_update_id)
           else:
              rewrite_file(last_update_id)
           tanggap(updates)
        time.sleep(5)

if __name__ == '__main__':
    main()
