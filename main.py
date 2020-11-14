# %% imports
import pandas as pd
from fazy import Fazy, Fang, FungSet
from IPython.display import display
from util import read_excel
import numpy as np
from matplotlib import pyplot as plt

# %% membaca data
data_mhs = read_excel('data/Mahasiswa.xls')
data_mhs.head()

# %% visualisasi data raw
data_mhs['Selisih'] = data_mhs['Penghasilan'] - data_mhs['Pengeluaran']
plt.xlabel('Penghasilan')
plt.ylabel('Pengeluaran')
plt.grid(True)
sct = plt.scatter(data_mhs['Penghasilan'], data_mhs['Pengeluaran'],
                  marker='o',
                  c=(data_mhs['Selisih'] < 0))
plt.legend(handles=sct.legend_elements()[0],
           labels=['Selisih positif', 'Selisih negatif'],
           loc='lower right')

# %% knowledge base
set_fungsi_penghasilan = (
  FungSet('kecil', Fang.linier_bawah, (4, 8)),
  FungSet('sedang', Fang.segitiga, (8, 9, 11)),
  FungSet('besar', Fang.linier_atas, (10, 20))
)

set_fungsi_pengeluaran = (
  FungSet('kecil', Fang.linier_bawah, (3, 6)),
  FungSet('sedang', Fang.segitiga, (5, 8, 10)),
  FungSet('besar', Fang.linier_atas, (9, 11)),
)

set_fungsi_kelayakan = (
  FungSet('rendah', Fang.linier_bawah, (20, 50)),
  FungSet('tinggi', Fang.linier_atas, (30, 60))
)

# %% visualisasi fungset
n_20 = np.arange(20, step=.1);
n_100 = np.arange(100, step=5);
h_pengh = np.max([fs.hitung(n_20) for fs in set_fungsi_penghasilan], axis=0)
h_penge = np.max([fs.hitung(n_20) for fs in set_fungsi_pengeluaran], axis=0)
h_kely = np.max([fs.hitung(n_100) for fs in set_fungsi_kelayakan], axis=0)

ylim = (0, 1.2)
plt.subplots_adjust(hspace=.4, top=1.5)
plt.subplot(3, 1, 1)
plt.ylim(ylim)
plt.title('Fungset Penghasilan')
plt.plot(n_20, h_pengh)
plt.subplot(3, 1, 2)
plt.ylim(ylim)
plt.title('Fungset Pengeluaran')
plt.plot(n_20, h_penge)
plt.subplot(3, 1, 3)
plt.ylim(ylim)
plt.title('Fungset Kelayakan')
plt.plot(n_100, h_kely)

#%% tabel inferensi
lookup_inferensi = pd.DataFrame({
  'Penghasilan': (
      'kecil', 'kecil', 'kecil',
      'sedang','sedang','sedang',
      'besar','besar','besar'),
  'Pengeluaran': (
      'kecil','sedang','besar',
      'kecil','sedang','besar',
      'kecil','sedang','besar'),
  'Kelayakan': (
      'rendah','tinggi','tinggi',
      'rendah','rendah','tinggi',
      'rendah','rendah','rendah')
  })

print('Tabel aturan inferensi')
lookup_inferensi

# %% main
model = Fazy((
    set_fungsi_penghasilan,
    set_fungsi_pengeluaran,
    set_fungsi_kelayakan
  ),
  lookup_inferensi=lookup_inferensi)

kelayakan = model.klasify(dataset=data_mhs, step=5) # (2, m)

# %% visualisasi tabel
data_mhs['Kelayakan'] = kelayakan
df1 = data_mhs.sort_values('Kelayakan', ascending=False)[:20]
# df2 = data_mhs.sort_values('Selisih', ascending=True)[:20]
print('Tabel terpilih terurut berdasarkan kelayakan')
display(df1)
# display(df2)

# %% output ke excel
with pd.ExcelWriter('data/Bantuan.xls') as writer:
  df1.to_excel(writer, 'data_bantuan', columns=['Id'], index=False, header=False)

# %% uji korelasi
df1.plot(y='Selisih', x='Kelayakan')
corr = data_mhs.loc[:, ['Kelayakan', 'Selisih']].corr().to_numpy()[0,1]
fitness = 1 - corr
print('Korelasi selisih-kelayakan', corr, sep='\n')
print('fitness:',fitness)