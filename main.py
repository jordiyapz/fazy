# %% imports
import pandas as pd
from fazy import Fazy, Fang, FungSet
from IPython.display import display
from util import read_excel
import numpy as np

# %% test
data_mhs = read_excel('data/Mahasiswa.xls')
data_mhs.head()

# %% knowledge base
set_fungsi_penghasilan = (
  FungSet('kecil', Fang.linier_bawah, (6, 10)),
  FungSet('sedang', Fang.trapesium, (9, 10, 14, 17)),
  FungSet('besar', Fang.linier_atas, (16, 18))
)

set_fungsi_pengeluaran = (
  FungSet('kecil', Fang.linier_bawah, (5, 7)),
  FungSet('sedang', Fang.trapesium, (6, 8, 9, 10)),
  FungSet('besar', Fang.linier_atas, (9, 11)),
)

set_fungsi_kelayakan = (
  FungSet('rendah', Fang.linier_bawah, (20, 50)),
  FungSet('tinggi', Fang.linier_atas, (30, 60))
)

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
      'tinggi','tinggi','tinggi',
      'rendah','tinggi','tinggi',
      'rendah','rendah','rendah')
  })

print('Tabel aturan inferensi')
lookup_inferensi

# %%
model = Fazy((
  set_fungsi_penghasilan,
  set_fungsi_pengeluaran,
  set_fungsi_kelayakan
  ),
  lookup_inferensi=lookup_inferensi)

kelayakan = model.klasify(dataset=data_mhs, step=1) # (2, m)

# %%
np.corrcoef(data_mhs['Penghasilan']-data_mhs['Pengeluaran'], kelayakan)[0,1]

# %%
data_mhs['Selisih'] = data_mhs['Penghasilan'] - data_mhs['Pengeluaran']
data_mhs['Kelayakan'] = kelayakan
df1 = data_mhs.sort_values('Kelayakan', ascending=False)[:20]
display(df1)
df2 = data_mhs.sort_values('Selisih', ascending=True)[:20]
display(df2)

# %% output ke excel
with pd.ExcelWriter('data/Bantuan.xls') as writer:
  df1.to_excel(writer, 'data_bantuan', columns=['Id'], index=False)

# %% uji korelasi
corr = data_mhs.loc[:, ['Kelayakan', 'Selisih']].corr().to_numpy()[0,1]
fitness = 1 - corr
print('fitness:',fitness)

# %%
df1.plot(y='Selisih', x='Kelayakan')
