# %% dependencies
import util
import numpy as np
import pandas as pd

# %% class set fungsi anggota
class FungSet:
  def __init__ (self, label, fungsi, bilangan:tuple):
    prev = bilangan[0]
    for bil in bilangan[1:]:
      assert prev <= bil, 'Bilangan harus terurut'
      prev = bil

    self.label = label
    self.fungsi = fungsi
    self.bilangan = bilangan

  def hitung(self, x, up=1):
    return self.fungsi(x, *self.bilangan, up=up)

# %% fungsi keanggotaan
class Fang:

  @staticmethod
  def linier_atas(x, a, b, up=1):
    return util.clamp((x - a) / (b - a), 0, up)

  @staticmethod
  def linier_bawah(x, a, b, up=1):
    return util.clamp((b - x) / (b - a), 0, up)

  @staticmethod
  def segitiga(x, a, b, c, up=1):
    if type(x) is np.ndarray:
      arr = np.where(x <= b,
        Fang.linier_atas(x, a, b, up),
        Fang.linier_bawah(x, b, c, up))
      return arr
    elif x <= b:
      return Fang.linier_atas(x, a, b, up)
    return Fang.linier_bawah(x, b, c, up)

  @staticmethod
  def trapesium(x, a, b, c, d, up=1):
    if type(x) is np.ndarray:
      arr = np.where(x <= c,
          Fang.linier_atas(x, a, b, up),
          Fang.linier_bawah(x, c, d, up))
      return arr
    elif x <= c:
      return Fang.linier_atas(x, a, b, up)
    return Fang.linier_bawah(x, c, d, up)

# %% class utamanya
class Fazy:
  def __init__(self, arr_fset:tuple, lookup_inferensi:pd.core.frame.DataFrame):
    # definisi notasi:
    # m     : banyaknya data (baris) di dataset
    # n     : banyaknya rules inferensi
    # c     : banyaknya kolom inferensi
    # nfs_i : banyaknya fungset di arr_fset di kolom ke-i

    for fset_tup in arr_fset:
      for fs in fset_tup:
        assert type(fs) == FungSet, \
          'arr_fset harus merupakan tuple berisi tuple FungSet'

    self.arr_fset = arr_fset

    # nama-nama kolom di inferensi
    self.cols = [col for col in lookup_inferensi]

    # one hot encoding setiap kategori di setiap kolom inferensi
    self.one_hot = tuple(
        pd.get_dummies(lookup_inferensi[col])[\
          [f.label for f in arr_fset[i]]] \
        for i, col in enumerate(self.cols))

  def _fazify(self, nilai:np.ndarray, fset_tup:tuple):
    return np.array([fs.hitung(nilai) for fs in fset_tup])

  def _inferensi(self, fazys):
    masked = tuple(
      np.dot(self.one_hot[i], fazys[i]) for i in range(len(fazys))
    )
    konjungsi = np.minimum(*masked)  # shape = (n, m)
    # matriks disjungsi
    disjungsi = np.dot(self.one_hot[-1].T, konjungsi) # shape = (nfs_out, m)
    return disjungsi

  def _defazify(self, inferensi, step, maks, mins):
    sumbu_x = np.arange(mins, maks, step=step)
    derajat = np.max([
      [fs.hitung(x, up=inferensi[i]) for x in sumbu_x] \
      for i, fs in enumerate(self.arr_fset[-1])
      ], axis=0)
    return np.dot(sumbu_x, derajat) / np.sum(derajat, axis=0)

  def klasify(self, dataset: pd.core.frame.DataFrame, step=10, maks=100, mins=0):
    fazys = tuple(  # fazys_i.shape = (nfs_i, m)
        self._fazify(dataset[self.cols[i]].to_numpy(), fset_tup) \
        for i, fset_tup in enumerate(self.arr_fset[:len(self.arr_fset)-1]))
    inferensi = self._inferensi(fazys)
    return self._defazify(inferensi, step, maks, mins)
