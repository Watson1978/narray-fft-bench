# CuPy との違い

CuPy 14.2.0 には `cupy.fft` の 18 関数と `cupyx.scipy.signal.fftconvolve` がすべてある。narray-linalg-bench の `cupy-gaps.md` と違って埋め合わせは要らないので、このファイルには挙動の差と実装の差を記録する。題材コードでは、ここに書いた差を踏まない。

| 項目 | CuPy | Cumo::FFT | 出どころ |
|---|---|---|---|
| 軸の範囲外 | 黙って通す | `Cumo::NArray::DimensionError` | cumo 側の計画 |
| 長さ 1 の `irfft`（既定の n が 0） | `CUFFT_INVALID_SIZE` | `ArgumentError` | cumo 側の計画 |
| `fftfreq` / `rfftfreq` | 割り算で、numpy と 1 ulp ずれる | numpy とビット一致 | cumo 側のセッション |
| `ifft`（2 の冪でない長さ） | 1 ulp ずれる（複素数の割り算の丸め） | 正しく丸める | cumo 側のセッション |
| cuFFT の作業領域 | プランごとに CuPy のメモリプールから確保し、プランがキャッシュにある間は持ち続ける。キャッシュは既定で 16 枚、メモリの上限なし（`memsize` が -1）。配列やプランの確保に失敗しても、キャッシュのプランは捨てない | プランは `cufftSetAutoAllocation(0)` で作り、作業領域は変換を呼ぶたびにメモリプールから借りて返す。プランの作成が `CUFFT_ALLOC_FAILED` なら、キャッシュのプランを捨て、GC とプールの解放をしてから作り直す | CuPy は D の n = 10⁷ fp64 で再現（下）。cumo は `ext/cumo/cuda/cufft.c`（e9ae5d76） |
| `fftconvolve` の変換の長さ | 各軸を、素因数が 2・3・5・7 だけの長さまで伸ばす（`cupyx.scipy.fft.next_fast_len`） | 同じ規則（`Cumo::FFT.fast_length`） | 両方のソース（CuPy 14.2.0、cumo e9ae5d76） |
| 要素ごとの演算のカーネルの読み込み | カーネルを 1 本ずつ、ディスクのキャッシュ（`~/.cupy/kernel_cache` の cubin）から読む。複素数の積の初回は 0.2 ms | 型ごとの演算の族を、CUDA の遅延読み込みで、族の最初の呼び出しのときに読む。1 族 3〜30 ms | D の初回（`docs/results/bignum_mul.md`）、cumo #583、#590、#610 |
| 複素数の `real` | ビューを返す | 新しい配列にコピーする（Numo と同じ） | D の `fft_pow2` の nsys |
| 複素数と実数の四則演算（`SComplex / SFloat` など） | 混ぜた型のカーネル 1 本（割り算なら `cupy_true_divide__complex64_float32_complex64`） | 0.14.0 までは、実数を複素数に変換するカーネル（`store_sfloat`）と演算のカーネルの 2 本（+ − × ÷ のどれも、どちらの順序でも）。0.14.1 で、同じ精度の組は 1 本（`div_r` など）になった（#632）。Ruby の Complex のスカラー、精度の違う組、整数と複素数は、まだ変換する（cumo 側のセッション） | E の nsys、[results/cumo-0.14.1.md](results/cumo-0.14.1.md) |
| 一時配列のメモリ | 参照がなくなるとすぐプールに戻り、次のステップで同じメモリを使う | Ruby の GC が回るまで戻らない（cumo #221、#222）。2²¹ 要素の SComplex の `zeros` を続けて呼ぶと Cumo 45 µs、CuPy 12 µs で、同じ配列の `fill(0)` なら両方 12.5 µs ほど。45 µs は、新しく確保した managed メモリへの最初の書き込みのページフォルト（cumo 側の説明） | D の速さの節 |

- CPU 側では、scipy 1.18.1 の `fftconvolve` が実数のとき 2・3・5 だけの長さへ伸ばし（`scipy.fft.next_fast_len(n, True)`）、numo-pocketfft 0.7.0 は伸ばさずに 2n − 1 のまま複素数の `fftn` で変換する。D の長さ 2n − 1 = 2·10ᵏ − 1 では、scipy、CuPy、cumo の 3 つとも 2·10ᵏ を選ぶ。

- 作業領域の再現（2026-10-17、RTX 5070 Ti Laptop の 12 GB、ほかのプロセスが 1.5 GB を使用）: D の n = 10⁷ fp64 で 5 経路を 1 プロセスで順に回すと、`rfft_exact`（長さ 19999999 の R2C と C2R）のあとに、キャッシュの 6 枚が 3.78 GiB を持ち、デバイスの空きは 0.41 GiB になった。次の `fft_pow2` で、配列の確保が `OutOfMemoryError` になり、プランの作成が `CUFFT_ALLOC_FAILED` になった。同じ 5 経路を、Cumo は n = 2×10⁷ まで 1 プロセスで回せた。D の関門では、CuPy の n1e7 と n2e7 だけ、経路ごとに別のプロセスで回した（`docs/results/bignum_mul.md`）。
- `import cupyx.scipy.signal` は、`cupyx.jit.rawkernel is experimental` の FutureWarning を標準エラーに出す（2026-10-17、CuPy 14.2.0）。動作には関係しない。
