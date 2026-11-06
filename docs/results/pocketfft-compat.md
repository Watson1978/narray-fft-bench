# Cumo::Pocketfft の互換

`Cumo::Pocketfft`（cumo master `e9ae5d76`）が、numo-pocketfft 0.7.0 の `Numo::Pocketfft` と同じように振る舞うかを確かめた。2026-10-18、numo-narray-alt 0.11.2、RTX 5070 Ti Laptop。試験は `test/test_pocketfft_parity.rb`（cumo が読めない環境では omit）。

## 比べたもの

- 13 関数: `fft`、`ifft`、`fft2`、`ifft2`、`fftn`、`ifftn`、`rfft`、`irfft`、`rfft2`、`irfft2`、`rfftn`、`irfftn`、`fftconvolve`。
- 入力の型: SFloat、DFloat、SComplex、DComplex、Int32。計画は 4 種としていたが、整数の配列を渡す使い方もあるので Int32 を足した。
- 入力の形: 1 次元（長さ 7、8、1、2）、2 次元（5 × 6、1 × 1、4 × 1）、3 次元（3 × 4 × 5）、空（[0]、[0, 3]）。`fftconvolve` は、長さ 7 と 8、1 と 1、1 と 5、1 × 1 どうし、3 × 4 と 2 × 5、2 × 3 × 4 と 3 × 2 × 2、次元数の違う組、空を含む組、0 次元どうしの 9 組を、型の 25 通りの組で回した。
- 値は `seq` の sin と cos から作った。乱数は使っていない。
- 比べた中身は、返る型と形、例外の型と文言（`Numo` を `Cumo` に読み替える）、値の相対誤差（max ノルム）。

## 結果

**0 次元の入力を `fftn` と `rfftn` に渡した場合を除いて、すべて一致した。** 総当たりの 887 ケースのうち、値を返したものが 415、両方が同じ例外を出したものが 462 だった。0 次元の 10 ケースは、下の「違い」に書く。

| 関数 | 受ける次元 | 返る型 | 複素数の入力 | 最後の軸の長さが 1 | 空、次元違い、NArray でない |
|---|---|---|---|---|---|
| `fft`、`ifft` | 1 | DComplex | 受ける | 値を返す | ArgumentError |
| `fft2`、`ifft2` | 2 | DComplex | 受ける | 値を返す | ArgumentError |
| `fftn`、`ifftn` | 1 以上 | DComplex | 受ける | 値を返す | ArgumentError（空と、NArray でない入力） |
| `rfft` | 1 | DComplex | CastError | 値を返す | ArgumentError |
| `rfft2` | 2 | DComplex | CastError | 値を返す | ArgumentError |
| `rfftn` | 1 以上 | DComplex | CastError | 値を返す | ArgumentError（空と、NArray でない入力） |
| `irfft` | 1 | DFloat | 受ける（実数も DComplex にしてから） | ShapeError | ArgumentError |
| `irfft2` | 2 | DFloat | 受ける | ShapeError | ArgumentError |
| `irfftn` | 1 以上 | DFloat | 受ける | ShapeError | ArgumentError（空と、NArray でない入力） |

- 例外の文言も一致する。ArgumentError は `Expect input array to be non-empty.`、`Expect input array to be one-dimensional.`、`Expect input array to be two-dimensional.`、`Expect class of input array to be Numo::NArray.`（cumo では `Cumo::NArray`）の 4 つ。CastError は `unknown conversion from Numo::DComplex to Numo::DFloat`（SComplex も同じ形）。ShapeError は `cannot get element of empty array`。
- `irfft` 系の出力の長さは、numo-pocketfft と同じく常に 2(m − 1) で、最後の軸の長さ m が 1 だと ShapeError になる。
- `fftconvolve`:
  - 返る型は、両方が実数（SFloat、DFloat、Int32 のどの組でも）なら DFloat、どちらかが複素数なら DComplex になる。
  - すべての軸の長さが 1 のとき（0 次元どうしを含む）は `a * b` を返すので、型は NArray の掛け算の規則に従う。SFloat どうしなら SFloat、Int32 どうしなら Int32、SComplex と SFloat なら SComplex になる。
  - 空を含む組は ArgumentError（`Expect input array to be non-empty.`）、次元数の違う組は ArgumentError（`Input arrays should have the same dimensionarity`。綴りも numo-pocketfft のまま）。
- **値の差:** 倍精度の結果では、相対誤差の最大が 3.49e-16（`irfftn`、Int32、3 × 4 × 5）、`fftconvolve` では 4.93e-16（DFloat と DComplex、3 次元）だった。単精度の結果（すべての軸が長さ 1 の `fftconvolve` が返す `a * b`）では、7.33e-8（SComplex どうし）だった。掛け算 1 回の丸めの差で、単精度の ε と同じ桁になる。tol は、それぞれ 10 倍の 5e-15 と 7.5e-7 にした。

## 違い（0 次元の入力）

| 関数 | numo-pocketfft | Cumo::Pocketfft |
|---|---|---|
| `fftn`、`rfftn`（実数の入力） | プロセスの状態で変わる（下） | DimensionError（`invalid axis (0 for 0-dimension)`） |
| `rfftn`（複素数の入力） | CastError | DimensionError |
| `ifftn`、`irfftn` | DimensionError（`invalid axis (0 for 0-dimension)`） | 同じ |
| 1 次元用、2 次元用の関数 | ArgumentError | 同じ |

- **numo-pocketfft の `fftn` と `rfftn` は、0 次元の配列を渡すと、配列の外を読む。** C の拡張（`ext/numo/pocketfft/pocketfftext.c`）が、変換の長さを `NA_SHAPE(x)[n_dims - 1]` で読み、`n_dims` が 0 のときも確かめないため。新しいプロセスで 1 回だけ呼ぶと、SFloat、DFloat、DComplex のどれでも NoMemoryError（`Failed to allocate memory for plan of pocketfft.`）になった。ほかの呼び出しのあとで呼ぶと、DFloat、SComplex、DComplex、Int32 で値を返したことがある。1 プロセスで型を変えながら続けて呼ぶと、120 秒たっても返らず、TERM でも止まらなかった（どの呼び出しで止まったかは確かめていない）。
- cumo は、これを写さずに DimensionError にしている（`lib/cumo/pocketfft.rb` のコメントに書いてある）。`ifftn` と `irfftn` の numo-pocketfft の例外と、型も文言も同じ。試験は、numo-pocketfft を 0 次元で呼ばず、cumo 側が DimensionError を出すことだけを確かめる。
- numo-pocketfft のこの不具合は、cumo の互換の問題ではないので、ここでは直さない（numo-pocketfft には報告しない。2026-10-18、ユーザーの判断）。

## 試験が違いを見つけるか

`Cumo::Pocketfft` に、わざと 4 種類の違いを入れて試験を回した（一時スクリプトで、試験の前に `prepend` した）。4 種類とも、試験が落ちた。

- `fft` の値を 1 + 1e-13 倍にする。
- `rfft` の返り値を SComplex にする。
- `ifft2` の例外の文言を変える。
- 1 次元の `fftconvolve` の最後の要素を落とす。
